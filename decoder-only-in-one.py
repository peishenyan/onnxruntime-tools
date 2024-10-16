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
onnx_output_dir = f"./logs/models/{path_name}"

os.makedirs(onnx_output_dir, exist_ok=True)

config = AutoConfig.from_pretrained(args.model)
config.use_cache = True
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    device_map="cpu",
).to(args.device).eval()

tokenizer = AutoTokenizer.from_pretrained(args.model)
text = "Give me a short introduction to large language model."
model_inputs = tokenizer(text, return_tensors="pt").to(args.device)

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
                position_ids=None,
                cache_position=None)-> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        # Pass input through the decoder model
        # decoder KV cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position)
            
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

def export_decoder(output_hidden_states, max_sequence_length, verbose=False, force_convert=False, export_static=True):
    """
    1. Create tokens for first inference model
    2. Only provide 4 tokens as input, no need of padding as we pad output kv cache later
    3. Output of this model will be next token and kv cache
    4. Export kv cache to explicit values
    """
    
    # input_ids = torch.zeros((1, max_sequence_length), dtype=input_ids_precision).to(args.device)
    # attention_mask = torch.zeros((1, max_sequence_length), dtype=mask_precision).to(args.device)
    
    input_ids = model_inputs.input_ids.to(args.device)
    attention_mask = model_inputs.attention_mask.to(args.device)
    init_len = input_ids.shape[1]
    print(init_len)
    
    # decoder_input = {'input_ids': model_inputs.input_ids.to(args.device),
    #                 'attention_mask': model_inputs.attention_mask.to(args.device)}
    # create decoder input for the first inference
    decoder_input = {'input_ids': padding_input(input_ids, max_sequence_length),
                     'attention_mask': padding_input(attention_mask, max_sequence_length)}

    # decoder_input['position_ids'] = torch.tensor([[init_len]], dtype=pos_ids_precision).to(args.device)
    decoder_input['position_ids'] = torch.arange(0, max_sequence_length, dtype=pos_ids_precision).unsqueeze(0).to(args.device)
    decoder_input['cache_position'] = torch.arange(0, max_sequence_length, dtype=pos_ids_precision).to(args.device)
    
    # run the decoder for the first inference
    
    with torch.no_grad():
        output = automodel_merged(input_ids=decoder_input['input_ids'],
                    attention_mask=decoder_input['attention_mask'])

    # extract past key values and logits for model export in step 3
    logits = output.logits
    past_key_values = output.past_key_values

    for i in range(128):
        new_token = torch.argmax(logits[0, i, :])
        print(new_token, tokenizer.decode([new_token], skip_special_tokens=False))

    for kv in past_key_values:
        kv[0][:,:,:,:] = 0
        kv[1][:,:,:,:] = 0

    print(past_key_values[0][0].shape)
    decoder_input['past_key_values'] = past_key_values
    decoder_input['attention_mask'] = padding_input(torch.ones((1, init_len), dtype=mask_precision).to(args.device), 2*max_sequence_length)

    with torch.no_grad():
        output = automodel_merged(input_ids=decoder_input['input_ids'],
                    attention_mask=decoder_input['attention_mask'],
                    past_key_values=decoder_input['past_key_values'],
                    position_ids=decoder_input['position_ids'],
                    cache_position=decoder_input['cache_position'])

    # extract past key values and logits for model export in step 3
    logits = output.logits
    past_key_values = output.past_key_values
    
    for kv in past_key_values:
        kv[0][:,:,:init_len,:] = kv[0][:,:,max_sequence_length:max_sequence_length+init_len,:]
        kv[1][:,:,:init_len,:] = kv[1][:,:,max_sequence_length:max_sequence_length+init_len,:]
        kv[0][:,:,init_len+1:,:] = 0
        kv[1][:,:,init_len+1:,:] = 0

    decoder_input['past_key_values'] = [[kv[0][:,:,:max_sequence_length,:], kv[1][:,:,:max_sequence_length,:]] for kv in past_key_values]

    for i in range(128):
        new_token = torch.argmax(logits[0, i, :])
        print(new_token, tokenizer.decode([new_token], skip_special_tokens=False))

    tokens = []
    new_token = torch.argmax(logits[0, -1, :])
    tokens.append(new_token)
    print(tokenizer.decode(tokens, skip_special_tokens=False))

    num_decoder_blocks = len(past_key_values)
    # each item in tuple is for specific layer, each inside tuple has key and value
    # if verbose:
    print(f"no of blocks: {num_decoder_blocks}, \
        past decoder key shape = {past_key_values[0][0].shape},\
        past decoder value shape = {past_key_values[0][1].shape}")

    # encoder output is static
    # starting seq length can be dynamic for both other inputs, we reshape model to static in base_model.py
    dynamic_axes = None
    decoder_onnx_path = f"{onnx_output_dir}/{path_name}_decoder_static_one_lm.onnx"

    # create input and output names for model conversion
    input_names = ['input_ids', 'attention_mask']
    input_names += sum([[f'past_key_values.{idx}.decoder.key',
              f'past_key_values.{idx}.decoder.value'
              ] for idx in range(num_decoder_blocks)], [])
    input_names += ['position_ids']
    output_names = ["logits"] + \
        sum([[f'present_key_values.{idx}.decoder.key',
              f'present_key_values.{idx}.decoder.value'
              ] for idx in range(int(num_decoder_blocks))], [])
    if verbose:
        print(f'input names for onnx model is {input_names}, output names for onnx model is {output_names}')



    with torch.no_grad():
        decoder_input['position_ids'] = torch.arange(init_len, init_len + max_sequence_length, dtype=pos_ids_precision).unsqueeze(0).to(args.device)
        decoder_input['attention_mask'] = padding_input(torch.ones((1, init_len+1), dtype=mask_precision).to(args.device), 2*max_sequence_length)
        for i in range(20):
            print(i)
            input_ids = torch.zeros((1,1), dtype=input_ids_precision)
            input_ids[0][0] = new_token
            decoder_input['input_ids'] = padding_input(input_ids, max_sequence_length)
            output = automodel_merged(input_ids=decoder_input['input_ids'],
                        attention_mask=decoder_input['attention_mask'],
                        past_key_values=decoder_input['past_key_values'],
                        position_ids=decoder_input['position_ids'])
            
            # decoder_input['position_ids'][0][0] += 1
            # print(decoder_input['position_ids'][0])
            past_key_values = output.past_key_values
            # decoder_input['past_key_values'] = [[torch.concat([kv[0][:,:,:init_len,:],kv[0][:,:,max_sequence_length:2*max_sequence_length-init_len,:]], dim=2),torch.concat([kv[1][:,:,:init_len,:],kv[1][:,:,max_sequence_length:2*max_sequence_length-init_len,:]], dim=2)] for kv in past_key_values]
            
            for kv in past_key_values:
                kv[0][:,:,init_len,:] = kv[0][:,:,max_sequence_length,:]
                kv[1][:,:,init_len,:] = kv[1][:,:,max_sequence_length,:]
                kv[0][:,:,init_len+1:,:] = 0
                kv[1][:,:,init_len+1:,:] = 0
            decoder_input['past_key_values'] = [[kv[0][:,:,:max_sequence_length,:], kv[1][:,:,:max_sequence_length,:]] for kv in past_key_values]
            decoder_input['attention_mask'][0][init_len+1] = 1
            init_len += 1
            decoder_input['position_ids'] = torch.arange(init_len, init_len + max_sequence_length, dtype=pos_ids_precision).unsqueeze(0).to(args.device)

            logits = output.logits
            new_token = torch.argmax(logits[0, 0, :])
            tokens.append(new_token)
            print(new_token, tokenizer.decode(tokens, skip_special_tokens=False))
        
    exit()

    # get onnx model path
    if force_convert:
        with torch.no_grad():
            print("Exporting merged_decoder!!!")
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

        # Checks
        model_onnx = onnx.load(decoder_onnx_path)  # load onnx model
        # onnx.checker.check_model(model_onnx)  # check onnx model
        # model_onnx, check = onnxsim.simplify(model_onnx)
        # assert check, "assert check failed"
        # onnx.save(model_onnx, decoder_onnx_path)

        if len(tied_params) > 0:
            remove_duplicate_weights_from_tied_info(
                model_onnx, automodel_merged, tied_params, save_path=decoder_onnx_path)
            

def pad_decoder_cache(past_key_values, padded_past_key_values, inf_iter, max_sequence_length, position_ids=0, cache_precision=torch.float32):
    if args.verbose:
        print('Before change', inf_iter, past_key_values[0][0].shape)
    # at the output of the first inference model, we perform left padding on kv cache
    past_key_values_modified = tuple()

    # perform padding for each attention head
    for kv in past_key_values:
        # create 0 padding of shape max sequence length - args.num_init_tokens (4 as kv cache 2nd dim = 4 due to 4 tokens)
        padded_key = torch.zeros((kv[0].shape[0], kv[0].shape[1], max_sequence_length, kv[0].shape[3]), dtype=cache_precision).to(args.device)
        padded_value = torch.zeros((kv[1].shape[0], kv[1].shape[1], max_sequence_length, kv[1].shape[3]), dtype=cache_precision).to(args.device)
        # # do right padding
        # padded_key = torch.cat((kv[0], nop_key), dim=2)
        # # print(padded_key.shape)
        # padded_value = torch.cat((kv[1], nop_value), dim=2)
        # no change to encoder kv cache
        past_key_values_modified += ((padded_key.to(args.device), padded_value.to(args.device)) ,)

    if args.verbose:
        print('After change', inf_iter, past_key_values_modified[0][0].shape)
    return past_key_values_modified

def create_decoder_attention_mask(attention_mask, inf_iter, max_sequence_length, position_ids=0, mask_precision=torch.int32, tokens=None):
    if args.verbose:
        print(f"after inference {inf_iter}, original mask size = {attention_mask.shape}")
    # after 1st inference, create attention mask of size max seq length - args.num_init_tokens as 4 tokens already computed on
    if inf_iter == 0:
        padded_mask = torch.zeros((1, max_sequence_length - args.num_init_tokens - 1), dtype=mask_precision).to(args.device)
        # to indicate valid attention on new token, we have to add 1 at the end
        attention_mask = torch.cat([attention_mask, padded_mask, torch.tensor([[1]]).to(args.device)], dim=1).to(mask_precision)
    else:
        # Update the mask at location = position id
        # if using 2d mask, we fill position id location with 1 else fill with 0
        # last element is already set to 1 for 2d mask and 0 for 4d mask to account for new token
        
        attention_mask[0, position_ids-1] = 1
    if args.verbose:
        print(f"after inference {inf_iter}, new mask size = {attention_mask.shape}")
    return attention_mask


def generate(max_sequence_length, use_past=True, use_static=True):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(args.device)

    # 指定 ONNX 模型的路径
    model_path = 'logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx'

    # 初始化 ONNX Runtime 会话
    session = ort.InferenceSession(model_path)


    # create list of tokens for english language and transcribe task, no need of time stamps
    tokens = []
    attention_mask = []
    # create decoder input for the first inference
    decoder_input = {'input_ids': model_inputs.input_ids.to(args.device),
                    'attention_mask': model_inputs.attention_mask.to(args.device)}

    t0 = time.time()

    for idx in range(max_sequence_length):
        # run the first inference which generates KV cache
        # if needed you can ignore kv cache and continue providing entire token sequence
        if idx == 0 or not use_past:
            with torch.no_grad():
                output = model.decoder(
                    input_ids=decoder_input['input_ids'],
                    attention_mask=decoder_input['attention_mask']
                    )
        else:
            # run 2nd and remaining token inference using last generated token, and kv cache
            # attention mask ie either None for dynamic model and indicates padded kv cache positions for static model
            # HACK: output_hidden_states is not actually uses for computation inside modeling_whipser file
            # only used for shape information, hence we test sending dummy value of same size as encoder output
            # NOTE: encoder hidden states refers to encoder output
            # comment out the next line if you still want to pass actual data
            decoder_input['position_ids'] = torch.tensor([idx + args.num_init_tokens - 1], dtype=torch.int32)
            with torch.no_grad():
                output = session.run([], input_feed={"input_ids":decoder_input['input_ids'],
                            "attention_mask":decoder_input['attention_mask'],
                            "position_ids":decoder_input['position_ids'],
                            "past_key_values":decoder_input['past_key_values']})

        # we need logits which is only present in conditional generation model
        logits = output.logits.detach().cpu() # shape [1, 4, 51865] after 1st inf
        # extract past key values
        past_key_values = output.past_key_values

        # why is this used? find out the token with highest probability
        new_token = np.argmax(logits[0, - 1, :])
        # print(f'new token is {tokenizer.decode(new_token, skip_special_tokens=True)}\n')

        # add new generated token to previous tokens
        tokens.append(new_token)
        
        # POST PROCESSING: the following code creates the decoder input for the next inference
        if not use_past:
            # use the concatenated tokens as input
            decoder_input['input_ids'] = torch.tensor([tokens], dtype=torch.int32).to(args.device)
        else:
            # only use the new token as input
            decoder_input['input_ids'] = torch.tensor([[new_token]], dtype=torch.int32).to(args.device)
            # for dynamic model, we don't need attention mask and we reuse past key values directly
            if not use_static:
                decoder_input['attention_mask'] = None
                decoder_input['past_key_values'] = past_key_values
            # for static model, we pad the decoder KV cache with 0's only after 1st inference and also create the proper attention mask
            else:
                if idx == 0:
                    padded_past_key_values = pad_decoder_cache(past_key_values, None,
                                                        idx, max_sequence_length)
                    decoder_input['past_key_values'] = padded_past_key_values
                else:    # for other inferences, we shift kv cache to left and modify the last element in place in the 3rd dimension
                    decoder_input['past_key_values'] = pad_decoder_cache(past_key_values, padded_past_key_values,
                                                                        idx, max_sequence_length)
                decoder_input['attention_mask'] = create_decoder_attention_mask(decoder_input['attention_mask'],
                                                                                idx, max_sequence_length)
        yield tokenizer.decode(tokens, skip_special_tokens=False), (time.time() - t0)

        if new_token == tokenizer.eos_token_id:
            break


if __name__ == "__main__":
    # evaluate()
    if args.export:
        print(f'Exporting Huggingface {path_name} to Onnx')
        export_decoder(None, max_sequence_length=args.length, force_convert=args.force_convert, export_static=args.static, verbose=False)
    else:
        print(f'Running Huggingface {path_name} inference')
        for decoded_sentence, elapsed in generate(max_sequence_length=args.length, use_past=True, use_static=args.static):
            print(f"{decoded_sentence}" , end="\r")
        print(f"\n\nElapsed time {elapsed:.2f} s")
