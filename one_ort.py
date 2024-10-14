import onnxruntime as ort
import argparse
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import login

def padding_input(input, max_sequence_length):
    print()
    input = np.concatenate([input, np.zeros(shape=(1,max_sequence_length-input.shape[1]), dtype=input.dtype)], axis=1)
    return input

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The model to convert", default='microsoft/Phi-3-mini-4k-instruct')
parser.add_argument('-q', '--quantize', type=int, help='whether use quantized INT4 or INT8 model', default=0)
args = parser.parse_args()
try:
    model = args.model.split('/')[1].replace('-', '_')
except:
    model = args.model.replace('-', '_')

if args.model == "microsoft/Phi-3-mini-4k-instruct":
    login(token='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')

device = "cpu" 
tokenizer = AutoTokenizer.from_pretrained(args.model)

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Which country is the largest over the world?"}, 
] 
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# prompt = "Which country is the largest over the world?"
input_data = tokenizer(prompt, return_tensors="pt")
input_ids = input_data.input_ids.numpy().astype(np.int32)
attention_mask = input_data.attention_mask.numpy().astype(np.int32)
print(input_ids)
print(attention_mask)
max_sequence_length = 128
start_len = input_ids.shape[1]
print('start_len', start_len)
print("Prompt is: ", prompt)

if args.quantize == 0:
    q_str = ''
elif args.quantize == 4:
    q_str = '_INT4'
elif args.quantize == 8:
    q_str = '_INT8'

model_path = 'logs/models/'+ model + '/' + model +'_decoder_static_one_lm'+q_str+'.onnx'

session = ort.InferenceSession(model_path)

input_names = []
output_names = []
for input in session.get_inputs():
    input_names.append(input.name)
for output in session.get_outputs():
    output_names.append(output.name)

tokens = []
sentence = []
### TEST
# position_ids = np.array([range(max_sequence_length,2*max_sequence_length)], dtype=np.int32)
position_ids = np.array([range(max_sequence_length)], dtype=np.int32)
inputs_1 = {"input_ids": padding_input(input_ids, max_sequence_length), "attention_mask": padding_input(np.concatenate([np.zeros((1,max_sequence_length), dtype=np.int32), np.ones((1,start_len), dtype=np.int32)], axis=1), 2*max_sequence_length), 'position_ids': position_ids}

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

kv_cache = {f'past_key_values.{i}.decoder.key': np.zeros((1, second_dim, max_sequence_length, forth_dim), dtype=np.float32) for i in range(num_layers)}
kv_cache.update({f'past_key_values.{i}.decoder.value': np.zeros((1, second_dim, max_sequence_length, forth_dim), dtype=np.float32) for i in range(num_layers)})
inputs_1.update(kv_cache)

outputs_1 = session.run(None, inputs_1)
logits = outputs_1[0]

# for i in range(128):
#     new_token = np.argmax(logits[0, i, :])
#     print(new_token, tokenizer.decode([new_token], skip_special_tokens=False))

new_token = np.argmax(logits[0, start_len-1, :])
tokens.append(new_token)
print(new_token, tokenizer.decode(tokens, skip_special_tokens=False))


input_ids = np.zeros((1,1), dtype=inputs_1['input_ids'].dtype)
attention_mask = np.ones((1,start_len+1), dtype=inputs_1['attention_mask'].dtype)
input_ids[0][0] = new_token
position_ids = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
inputs_2 = {"input_ids": padding_input(input_ids, max_sequence_length), "attention_mask": padding_input(attention_mask, 2*max_sequence_length), "position_ids": position_ids}
for layer in range(num_layers):
    kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,:start_len,:] = outputs_1[2*layer+1][:,:,max_sequence_length:max_sequence_length+start_len,:]
    kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,:start_len,:] = outputs_1[2*layer+2][:,:,max_sequence_length:max_sequence_length+start_len,:]
inputs_2.update(kv_cache)

# print('kv_cache', outputs_1[2*layer+1][:,:,0,:], outputs_1[2*layer+1][:,:,max_sequence_length-1,:], outputs_1[2*layer+1][:,:,max_sequence_length,:], outputs_1[2*layer+1][:,:,max_sequence_length+1,:])


for i in range(max_sequence_length // 2):
    input_ids = np.zeros((1,1), dtype=inputs_1['input_ids'].dtype)
    input_ids[0][0] = new_token
    inputs_2['input_ids'] =  padding_input(input_ids, max_sequence_length)

    outputs = session.run(None, inputs_2)
    for layer in range(num_layers):
        kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len,:] = outputs[2*layer+1][:,:,max_sequence_length,:]
        kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len,:] = outputs[2*layer+2][:,:,max_sequence_length,:]
        kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len+1:,:] = 0
        kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len+1:,:] = 0

    inputs_2.update(kv_cache)
    # print('kv_cache', inputs_2['past_key_values.0.decoder.key'][0,0,start_len,0], inputs_2['past_key_values.0.decoder.key'][0,0,start_len+1,0])
    inputs_2["attention_mask"][0][start_len+1] = 1
    start_len += 1
    inputs_2["position_ids"] = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
    
    logits = outputs[0]
    new_token = np.argmax(logits[0, 0, :])
    tokens.append(new_token)
    print(tokenizer.decode(tokens, skip_special_tokens=False))

    if new_token == tokenizer.eos_token_id:
        break

print('--------------------Done.-------------------------')


prompt = "So how many people are there?"
print('The second question is:', prompt)
input_data = tokenizer(prompt, return_tensors="pt")
input_ids = input_data.input_ids.numpy().astype(np.int32)
second_len = input_ids.shape[1]
position_ids = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
inputs_2["attention_mask"][0][start_len] = 0
inputs_2["attention_mask"][0][max_sequence_length:max_sequence_length+second_len] = 1
inputs_2["input_ids"] = padding_input(input_ids, max_sequence_length)
inputs_2['position_ids'] = position_ids

outputs = session.run(None, inputs_2)

for layer in range(num_layers):
    kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len:start_len+second_len,:] = outputs[2*layer+1][:,:,max_sequence_length:max_sequence_length+second_len,:]
    kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len:start_len+second_len,:] = outputs[2*layer+2][:,:,max_sequence_length:max_sequence_length+second_len,:]

start_len += second_len
inputs_2.update(kv_cache)
# print('kv_cache', inputs_2['past_key_values.0.decoder.key'][0,0,start_len,0], inputs_2['past_key_values.0.decoder.key'][0,0,start_len+1,0])
inputs_2["attention_mask"][0][start_len+1] = 1
start_len += 1
inputs_2["position_ids"] = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)

logits = outputs[0]
new_token = np.argmax(logits[0, second_len-1, :])
tokens.append(new_token)
print(tokenizer.decode(tokens, skip_special_tokens=False))

input_ids = np.zeros((1,1), dtype=inputs_1['input_ids'].dtype)
input_ids[0][0] = new_token
inputs_2['input_ids'] =  padding_input(input_ids, max_sequence_length)
attention_mask = np.ones((1,start_len+1), dtype=inputs_1['attention_mask'].dtype)
inputs_2["attention_mask"] = padding_input(attention_mask, 2*max_sequence_length)

for i in range(max_sequence_length // 2):

    outputs = session.run(None, inputs_2)
    for layer in range(num_layers):
        kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len,:] = outputs[2*layer+1][:,:,max_sequence_length,:]
        kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len,:] = outputs[2*layer+2][:,:,max_sequence_length,:]
        kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len+1:,:] = 0
        kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len+1:,:] = 0

    inputs_2.update(kv_cache)
    # print('kv_cache', inputs_2['past_key_values.0.decoder.key'][0,0,start_len,0], inputs_2['past_key_values.0.decoder.key'][0,0,start_len+1,0])
    inputs_2["attention_mask"][0][start_len+1] = 1
    start_len += 1
    inputs_2["position_ids"] = np.array([range(start_len, start_len+max_sequence_length)], dtype=np.int32)
    
    logits = outputs[0]
    new_token = np.argmax(logits[0, 0, :])
    tokens.append(new_token)
    print(tokenizer.decode(tokens, skip_special_tokens=False))

    input_ids = np.zeros((1,1), dtype=inputs_1['input_ids'].dtype)
    input_ids[0][0] = new_token
    inputs_2['input_ids'] =  padding_input(input_ids, max_sequence_length)

    if new_token == tokenizer.eos_token_id:
        break

print(tokenizer.decode(tokens, skip_special_tokens=False))