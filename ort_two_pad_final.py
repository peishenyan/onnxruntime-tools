import onnxruntime as ort
import argparse
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import login
import os

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

model_path_1 = './logs/models/done/1_prefill_INT4_final.onnx'
model_path_2 = './logs/models/done/2_decode_INT4_final.onnx'


session_1 = ort.InferenceSession(model_path_1)
session_2 = ort.InferenceSession(model_path_2)

input_names = []
output_names = []
for input in session_1.get_inputs():
    input_names.append(input.name)
for output in session_1.get_outputs():
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

questions = [
    [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": "Write an article about spring with 500 words."}],
    # [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": "What is the result of one plus one?"}],
    # [{"role": "user", "content": "What is the result of multiplying the last result with two?"}],
    # [{"role": "user", "content": 'What is the result of subtracting latest result with three?'}],
    # [{"role": "user", "content": 'Is the final result equal to one?'}],
]
messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "What is the result of one plus one?"}, 
] 
prompt_list = []
for messages in questions:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_list.append(prompt)

input_data = tokenizer(prompt, return_tensors="pt")
input_ids = input_data.input_ids.numpy().astype(np.int32)
attention_mask = input_data.attention_mask.numpy().astype(np.int32)

max_sequence_length = 128
max_cache_length = 256
start_len = 0

sentence = []

input = {}
kv_cache = {f'past_key_values.{i}.decoder.key': np.zeros((1, second_dim, max_cache_length, forth_dim), dtype=np.float32) for i in range(num_layers)}
kv_cache.update({f'past_key_values.{i}.decoder.value': np.zeros((1, second_dim, max_cache_length, forth_dim), dtype=np.float32) for i in range(num_layers)})

for question in prompt_list:
    tokens = []
    input.update(kv_cache)
    sentence.append(question)
    os.system('cls' if os.name == 'nt' else 'clear')
    for s in sentence:
        print(s)
    input_data = tokenizer(question, return_tensors="pt")
    input_ids = input_data.input_ids.numpy().astype(np.int32)
    input_len = input_ids.shape[1]
    position_ids = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
    attention_mask = None
    if start_len == 0:
        attention_mask = np.zeros((1,max_cache_length), dtype=np.int32)
    else:
        attention_mask = np.ones((1,min(start_len,max_cache_length)), dtype=np.int32)

    input["attention_mask"] = padding_input(padding_input_reverse(attention_mask, max_cache_length), max_cache_length+max_sequence_length)
    input["attention_mask"][0][max_cache_length:max_cache_length+input_len] = 1
    input["input_ids"] = padding_input(input_ids, max_sequence_length)
    input['position_ids'] = position_ids

    outputs = session_1.run(None, input)
    
    for layer in range(num_layers):
        kv_cache[f'past_key_values.{layer}.decoder.key'] = outputs[2*layer+1]
        kv_cache[f'past_key_values.{layer}.decoder.value'] = outputs[2*layer+2]

    start_len += input_len
    input.update(kv_cache)
    
    new_token = outputs[0][0][0]
    tokens.append(new_token)
    os.system('cls' if os.name == 'nt' else 'clear')
    for s in sentence:
        print(s)
    print(tokenizer.decode(tokens, skip_special_tokens=False))

    while True:
        input_ids = np.zeros((1,1), dtype=np.int32)
        input_ids[0][0] = new_token
        input['input_ids'] = input_ids

        attention_mask = np.ones((1,min(start_len, max_cache_length)), dtype=input['attention_mask'].dtype)
        input["attention_mask"] = padding_input(padding_input_reverse(attention_mask, max_cache_length), max_cache_length+1)
        input['attention_mask'][0][max_cache_length] = 1

        position_ids = np.array([[start_len]], dtype=np.int32)
        input["position_ids"] = position_ids
        
        outputs = session_2.run(None, input)
        for layer in range(num_layers):
            kv_cache[f'past_key_values.{layer}.decoder.key'] = outputs[2*layer+1]
            kv_cache[f'past_key_values.{layer}.decoder.value'] = outputs[2*layer+2]

        input.update(kv_cache)
        start_len += 1
        
        new_token = outputs[0][0][0]
        tokens.append(new_token)
        os.system('cls' if os.name == 'nt' else 'clear')
        answers = tokenizer.decode(tokens, skip_special_tokens=False)
        for s in sentence:
            print(s)
        print(answers)
        if new_token == tokenizer.eos_token_id:
            break

    sentence.append(answers)
