import onnxruntime as ort
import argparse
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import login

def padding_input(input, max_sequence_length):
    if input.shape[1] >= max_sequence_length:
        return input
    input = np.concatenate([input, np.zeros(shape=(1,max_sequence_length-input.shape[1]), dtype=input.dtype)], axis=1)
    return input

def padding_input_reverse(input, max_sequence_length):
    if input.shape[1] >= max_sequence_length:
        return input
    input = np.concatenate([np.zeros(shape=(1,max_sequence_length-input.shape[1]), dtype=input.dtype), input], axis=1)
    return input

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The model to convert", default='microsoft/Phi-3-mini-4k-instruct')
parser.add_argument('-q', '--quantize', type=int, help='whether use quantized INT4 or INT8 model', default=0)
args = parser.parse_args()
try:
    model = args.model.split('/')[1].replace('-', '_')
except:
    model = args.model.replace('-', '_')


device = "cpu" 
tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.quantize == 0:
    q_str = ''
elif args.quantize == 4:
    q_str = '_INT4'
elif args.quantize == 8:
    q_str = '_INT8'

model_path = 'logs/models/pad/'+ model + '/done/' + model +'_decoder_static_one_lm'+q_str+'_ex_v21_INT4_QDQ.onnx'
session = ort.InferenceSession(model_path)

input_names = []
output_names = []
for input in session.get_inputs():
    input_names.append(input.name)
for output in session.get_outputs():
    output_names.append(output.name)

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

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "What is the result of one plus one?"}, 
] 
prompt_list = []
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_list.append(prompt)
prompt_list.append('What is the result of multiplying the last result with two?')
prompt_list.append('What is the latest result minus three?')
prompt_list.append('Is the final result equal to one?')

max_sequence_length = 128
start_len = 0
print('start_len', start_len)
print("Prompt is: ", prompt)

tokens = []
sentence = []

input = {}
kv_cache = {f'past_key_values.{i}.decoder.key': np.zeros((1, second_dim, max_sequence_length, forth_dim), dtype=np.float32) for i in range(num_layers)}
kv_cache.update({f'past_key_values.{i}.decoder.value': np.zeros((1, second_dim, max_sequence_length, forth_dim), dtype=np.float32) for i in range(num_layers)})
input.update(kv_cache)

for question in prompt_list:
    print('This question is: ', question)
    input_data = tokenizer(question, return_tensors="pt")
    input_ids = input_data.input_ids.numpy().astype(np.int32)
    input_len = input_ids.shape[1]
    position_ids = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
    attention_mask = None
    if start_len == 0:
        attention_mask = np.zeros((1,max_sequence_length), dtype=np.int32)
    else:
        attention_mask = np.ones((1,start_len), dtype=np.int32)
    
    input["attention_mask"] = padding_input(padding_input_reverse(attention_mask, max_sequence_length), 2*max_sequence_length)
    input["attention_mask"][0][max_sequence_length:max_sequence_length+input_len] = 1
    input["input_ids"] = padding_input(input_ids, max_sequence_length)
    input['position_ids'] = position_ids


    outputs = session.run(None, input)

    for layer in range(num_layers):
        kv_cache[f'past_key_values.{layer}.decoder.key'] = outputs[2*layer+1]
        kv_cache[f'past_key_values.{layer}.decoder.value'] = outputs[2*layer+2]

    start_len += input_len
    start_len = min(start_len,max_sequence_length)
    input.update(kv_cache)

    logits = outputs[0]
    new_token = np.argmax(logits[0, input_len-1, :])
    tokens.append(new_token)
    print(tokenizer.decode(tokens, skip_special_tokens=False))

    
    for i in range(max_sequence_length):
        input_ids = np.zeros((1,1), dtype=input['input_ids'].dtype)
        input_ids[0][0] = new_token
        input['input_ids'] =  padding_input(input_ids, max_sequence_length)

        attention_mask = np.ones((1,start_len), dtype=input['attention_mask'].dtype)
        input["attention_mask"] = padding_input(padding_input_reverse(attention_mask, max_sequence_length), 2*max_sequence_length)
        input['attention_mask'][0][max_sequence_length] = 1

        position_ids = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
        input["position_ids"] = position_ids

        outputs = session.run(None, input)
        for layer in range(num_layers):
            kv_cache[f'past_key_values.{layer}.decoder.key'] = outputs[2*layer+1]
            kv_cache[f'past_key_values.{layer}.decoder.value'] = outputs[2*layer+2]

        input.update(kv_cache)
        start_len += 1
        start_len = min(start_len,max_sequence_length)
        
        logits = outputs[0]
        new_token = np.argmax(logits[0, 0, :])
        tokens.append(new_token)
        print(tokenizer.decode(tokens, skip_special_tokens=False))

        if new_token == tokenizer.eos_token_id:
            break


    print('--------------------Done.-------------------------')
