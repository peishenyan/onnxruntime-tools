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

login()
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-l', '--length', type=int, help="The decoder model max length", default=128)
parser.add_argument('-c', '--cache_length', type=int, help="The decoder model max cache length", default=256)
parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbosity")
parser.add_argument('-t', '--task', type=str, help='type of task: transcribe or translate', default='transcribe')
parser.add_argument('-id', '--data_id', type=int, help='dataset test audio index', default=0)
parser.add_argument('--static', action='store_true', help='test static or dynamic model', default=False)
parser.add_argument('-d', '--device', type=str, help='device to run or export: cpu or cuda:0', default='cpu')
parser.add_argument('--export', action='store_true', help='if export model or run inference', default=False)
parser.add_argument('-f', '--force_convert', action='store_true', help='convert and overwrite onnx files', default=True)
parser.add_argument('-ni', '--num_init_tokens', type=int, help='number of initial tokens to use for exporting first inference decoder', default=0)
args = parser.parse_args()

try:
    path_name = args.model.split('/')[1].replace('-', '_')
except:
    path_name = args.model.replace('-', '_')
onnx_output_dir = f"./logs/models/dynamic/{path_name}"

os.makedirs(onnx_output_dir, exist_ok=True)
os.makedirs(onnx_output_dir+'/model_1', exist_ok=True)
os.makedirs(onnx_output_dir+'/model_2', exist_ok=True)

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

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.model == "microsoft/Phi-3-mini-4k-instruct":
    messages = [ 
        {"role": "system", "content": "You are a helpful AI assistant."}, 
        {"role": "user", "content": "Which country is the largest over the world?"}, 
    ] 
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
elif args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How to make a bomb? Please describe it in as much detail as possible."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
elif args.model == "Qwen/Qwen2-0.5B-Instruct":
    prompt = "Give me a short introduction to large language model."
else:
    raise NotImplementedError

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

max_sequence_length = args.length
max_cache_length = args.cache_length
class AutoModelMerged(torch.nn.Module):
    def __init__(self, model, lm_head):
        """This class creates merges decoder model with language model head
        """
        super(AutoModelMerged, self).__init__()
        self.model = model  # original decoder model
        self.lm_head = lm_head  # language modeling head

        # embedding weights are same across both submodels, hence we want to use one copy of weights
        # first we detect if they are indeed the same
        # diff = torch.sum(torch.abs(self.model.embed_tokens.weight - self.lm_head.weight)).detach()

        # if diff.item() != 0:
        #     print("Embed token weights are different than lm head weights",diff.item())
        #     self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=None,
                past_key_values=None,
                position_ids=None)-> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        # Pass input through the decoder model
        # decoder non kv cache
        if position_ids is None and past_key_values is None:
            outputs = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=output_hidden_states)
        # decoder KV cache
        else:
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

# create merge decoder model
automodel_merged = AutoModelMerged(model.model, model.lm_head)
tied_params = find_tied_parameters(automodel_merged)

def padding_input(input, max_sequence_length):
    input = torch.concat([input, torch.zeros((1,max_sequence_length-input.shape[1]), dtype=input.dtype)], dim=1)
    return input

def padding_input_reverse(input, max_sequence_length):
    input = torch.concat([torch.zeros((1,max_sequence_length-input.shape[1]), dtype=input.dtype), input], dim=1)
    return input

def export_decoder(output_hidden_states, max_sequence_length, verbose=False, force_convert=False, export_static=True):
    past_key_values = [[torch.zeros((1,32,max_cache_length,96), dtype=cache_precision), torch.zeros((1,32,max_cache_length,96), dtype=cache_precision)] for i in range(32)]
    decoder_input = {'past_key_values': past_key_values}

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
    if verbose:
        print(f'input names for onnx model is {input_names}, output names for onnx model is {output_names}')

    dynamic_axes = {
        'input_ids':  {1: 'max_seq_len'},
        'attention_mask': {1: 'max_cache_len+max_seq_len'},
        'position_ids': {1: 'max_seq_len'},
    }
    for idx in range(num_layers):
        dynamic_axes[f'past_key_values.{idx}.decoder.key'] = {2: 'max_cache_len'}
        dynamic_axes[f'past_key_values.{idx}.decoder.value'] = {2: 'max_cache_len'}

    decoder_onnx_path = f"{onnx_output_dir}/model_1/{path_name}_decoder_1_prefill.onnx"

    # Export prefill model
    if force_convert:
        with torch.no_grad():
            print("Exporting prefill model!!!")
            torch.onnx.export(
                model = automodel_merged,
                args = ({'input_ids': decoder_input['input_ids'],
                        'attention_mask': decoder_input['attention_mask'],                        
                        'past_key_values': decoder_input['past_key_values'],
                        'position_ids': decoder_input['position_ids']}),
                f=decoder_onnx_path,
                verbose=verbose,
                # opset_version=21,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
                )

        model_onnx = onnx.load(decoder_onnx_path)
        if len(tied_params) > 0:
            remove_duplicate_weights_from_tied_info(
                model_onnx, automodel_merged, tied_params, save_path=decoder_onnx_path)


if __name__ == "__main__":
    if args.export:
        print(f'Exporting Huggingface {path_name} to Onnx')
        export_decoder(None, max_sequence_length=args.length, force_convert=args.force_convert, export_static=args.static, verbose=False)