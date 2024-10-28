'''
Two stages of a decoder-only model generation:
1. Prefill: Process user's input, and generate the first token
2. Decode: Continue to generate with the last token as input

How to generate an ONNX model for decoder-only model with static kv cache:

We should first deterfine the max _cache_length (the maximum length of each kv cache to memorize past kv) and the max_seq_length (the maximum length of user's input tokens).
So for the prefill stage, the input shape ==> input_ids: [1, max_seq_length], attention_mask: [1, max_seq_length+max_cache_length], position_ids: [1, max_seq_length], past_key_values.{i}.key/value: [1, x, max_cache_length, y]
output shape ==> logits: [1, max_seq_length, total_num_tokens], present_key_values.{i}.key/value: [1, x, max_cache_length+max_seq_length, y]
And for the decode stage, input_ids: [1, 1], attention_mask: [1, 1+max_cache_length], position_ids: [1, 1], past_key_values.{i}.key/value: [1, x, max_cache_length, y]
output shape ==> logits: [1, 1, total_num_tokens], present_key_values.{i}.key/value: [1, x, max_cache_length+1, y]

Now let's begin with how kv cache works and go through how to pad our inputs. 
The attention mask has two parts: the left part is for the past kv cache, and the right part is for the input_ids.
We set the mask value to one for those real kv caches and input_ids, and set the mask value to zero for those padded things.

For each key / value cache, imagine that there are max_cache_length slots to store the key / value of tokens. 
We store the past key and values at the end of cache slots, which can be seen as left padding.
Then, we set the attention mask to zero to ignore the padded kv. 
For static input length, we use right padding, and also set the attention mask to zero to ignore the padded tokens.

To accelerate the decoding process, the input length of the decode stage should be one to reduce the computational cost. We call this disaggregation of prefiling and decoding.
According to this idea, we generate the corresponding inputs and use torch.onnx.export to export the ONNX model, with dynamic axes for max_cache_length and max_input_length. 

The following steps are how to quantize the model and align the output with the input for kv cache.
Google protobuf limits the maximum size of onnx model file, so the first step is to convert the initializers of model to external data. 
Then, due to the matmul_4bits_quantizer may modify some ops in opset version 21, we should change the model to v21. After that, we use quantizer to quantize the model to INT4.
Finally, we align the input with output of the kv cache. According to the padding mode we introduced before, we can just slice the present_key_value cache.

For running time, we only need one onnx model and load it twice with different freedimOverride (max_input_len=1)
'''


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Phi3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
import argparse
import torch
import time
import os
import onnxruntime as ort
import onnx
from typing import Tuple, Union
from accelerate.utils import find_tied_parameters
from optimum.onnx import remove_duplicate_weights_from_tied_info
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-l', '--length', type=int, help="The decoder model max length", default=128)
parser.add_argument('-c', '--cache_length', type=int, help="The decoder model max cache length", default=256)
parser.add_argument('-d', '--device', type=str, help='device to run or export: cpu or cuda:0', default='cpu')
parser.add_argument('--decode', action='store_true', help='whether the model is for decode or prefill', default=False)
args = parser.parse_args()

try:
    path_name = args.model.split('/')[1].replace('-', '_')
except:
    path_name = args.model.replace('-', '_')
onnx_output_dir = f"./logs/models/{path_name}"

os.makedirs(onnx_output_dir, exist_ok=True)

config = AutoConfig.from_pretrained(args.model)
config.use_cache = True
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    device_map="cpu",
).to(args.device).eval()

if args.model == "microsoft/Phi-3-mini-4k-instruct":
    num_layers = 32 
    second_dim = 32
    forth_dim = 96
elif args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
    num_layers = 22
    second_dim = 4
    forth_dim = 64
elif args.model == "Qwen/Qwen2-0.5B-Instruct":
    num_layers = 24
    second_dim = 2
    forth_dim = 64
else:
    raise NotImplementedError

tokenizer = AutoTokenizer.from_pretrained(args.model)
messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Which country is the largest over the world?"}, 
] 
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

if "cuda" in args.device:
    model.half()
    encoder_input_precision = torch.float16
    input_ids_precision = torch.int32
    mask_precision = torch.float16
    pos_ids_precision = torch.int32
    cache_precision = torch.float16
else:
    encoder_input_precision = torch.float32
    input_ids_precision = torch.int32
    mask_precision = torch.int32
    pos_ids_precision = torch.int32
    cache_precision = torch.float32

class AutoModel(torch.nn.Module):
    def __init__(self, model, lm_head):
        super(AutoModel, self).__init__()
        self.model = model  # original decoder model
        self.lm_head = lm_head  # language modeling head

    def forward(self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=None,
                past_key_values=None,
                position_ids=None)-> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        # Pass input through the decoder model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids)
            
        # Pass the output through the lm head
        lm_logits = self.lm_head(outputs[0])
        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values
        )

# create decoder model
automodel = AutoModel(model.model, model.lm_head)
tied_params = find_tied_parameters(automodel)

def padding_input(input, max_sequence_length):
    input = torch.concat([input, torch.zeros((1,max_sequence_length-input.shape[1]), dtype=input.dtype)], dim=1)
    return input

def padding_input_reverse(input, max_sequence_length):
    input = torch.concat([torch.zeros((1,max_sequence_length-input.shape[1]), dtype=input.dtype), input], dim=1)
    return input

def export_decoder(max_sequence_length, max_cache_length):
    past_key_values = [[torch.zeros((1,second_dim,max_cache_length,forth_dim), dtype=cache_precision), torch.zeros((1,second_dim,max_cache_length,forth_dim), dtype=cache_precision)] for i in range(num_layers)]
    decoder_input = {'past_key_values': past_key_values}

    # I. Prefill
    if args.decode == False:
        input_ids = model_inputs.input_ids.to(args.device).to(input_ids_precision)
        init_len = input_ids.shape[1]

        # create decoder input for the first inference
        decoder_input['input_ids'] = padding_input(input_ids, max_sequence_length)
        decoder_input['position_ids'] = torch.arange(0, max_sequence_length, dtype=pos_ids_precision).unsqueeze(0).to(args.device)
        decoder_input['attention_mask'] = padding_input_reverse(padding_input(torch.ones((1, init_len), dtype=mask_precision).to(args.device), max_sequence_length), max_cache_length+max_sequence_length)


        input_names = ['input_ids', 'attention_mask']
        input_names += sum([[f'past_key_values.{idx}.decoder.key',
                f'past_key_values.{idx}.decoder.value'
                ] for idx in range(num_layers)], [])
        input_names += ['position_ids']
        output_names = ["logits"] + \
            sum([[f'present_key_values.{idx}.decoder.key',
                f'present_key_values.{idx}.decoder.value'
                ] for idx in range(int(num_layers))], [])
        
        print(f'input names for onnx model is {input_names}, output names for onnx model is {output_names}')

        dynamic_axes = None
        decoder_onnx_path = f"{onnx_output_dir}/{path_name}_decoder_1_prefill.onnx"

        # Export prefill model
        with torch.no_grad():
            print("Exporting prefill model!!!")
            torch.onnx.export(
                model = automodel,
                args = ({'input_ids': decoder_input['input_ids'],
                        'attention_mask': decoder_input['attention_mask'],                        
                        'past_key_values': decoder_input['past_key_values'],
                        'position_ids': decoder_input['position_ids']}),
                f=decoder_onnx_path,
                # opset_version=21, # now v21 is not available
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
                )

        model_onnx = onnx.load(decoder_onnx_path)
        if len(tied_params) > 0:
            remove_duplicate_weights_from_tied_info(
                model_onnx, automodel, tied_params, save_path=decoder_onnx_path)

    else:
        # II. Decoding
        init_len = 1
        decoder_input['position_ids'] = torch.tensor([[init_len]], dtype=pos_ids_precision).to(args.device)
        decoder_input['attention_mask'] = padding_input_reverse(torch.ones((1, init_len+1), dtype=mask_precision).to(args.device), max_cache_length+1)
        input_ids = torch.zeros((1,1), dtype=input_ids_precision)
        decoder_input['input_ids'] = input_ids
        
        input_names = ['input_ids', 'attention_mask']
        input_names += sum([[f'past_key_values.{idx}.decoder.key',
                f'past_key_values.{idx}.decoder.value'
                ] for idx in range(num_layers)], [])
        input_names += ['position_ids']
        output_names = ["logits"] + \
            sum([[f'present_key_values.{idx}.decoder.key',
                f'present_key_values.{idx}.decoder.value'
                ] for idx in range(int(num_layers))], [])
        
        dynamic_axes = None
        decoder_onnx_path = f"{onnx_output_dir}/{path_name}_decoder_2_decode.onnx"

        # Export decode model
        with torch.no_grad():
            print("Exporting decode model!!!")
            torch.onnx.export(
                model = automodel,
                args = ({'input_ids': decoder_input['input_ids'],
                        'attention_mask': decoder_input['attention_mask'],                        
                        'past_key_values': decoder_input['past_key_values'],
                        'position_ids': decoder_input['position_ids']}),
                f=decoder_onnx_path,
                # opset_version=21,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
                )

        model_onnx = onnx.load(decoder_onnx_path)
        if len(tied_params) > 0:
            remove_duplicate_weights_from_tied_info(
                model_onnx, automodel, tied_params, save_path=decoder_onnx_path)

if __name__ == "__main__":
    print(f'Exporting Huggingface {path_name} to Onnx')
    export_decoder(max_sequence_length=args.length, max_cache_length=args.cache_length)