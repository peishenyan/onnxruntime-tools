import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

def padding_input(input, max_sequence_length):
    input = np.concatenate([input, np.zeros(shape=(1,max_sequence_length-input.shape[1]), dtype=input.dtype)], axis=1)
    return input


device = "cpu" 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
prompt = "Give me a short introduction to large language model."
input_data = tokenizer(prompt, return_tensors="pt")
input_ids = input_data.input_ids.numpy().astype(np.int32)
attention_mask = input_data.attention_mask.numpy().astype(np.int32)

max_sequence_length = 128
start_len = input_ids.shape[1]

model_path_1 = 'logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_non_kvcache_lm.onnx'
model_path_2 = 'logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx'

session_1 = ort.InferenceSession(model_path_1)
session_2 = ort.InferenceSession(model_path_2)

input_names = []
output_names = []
for input in session_1.get_inputs():
    input_names.append(input.name)
for output in session_1.get_outputs():
    output_names.append(output.name)

tokens = []
sentence = []
inputs_1 = {"input_ids": padding_input(input_ids, max_sequence_length), "attention_mask": padding_input(attention_mask, max_sequence_length)}
outputs_1 = session_1.run(None, inputs_1)


logits = outputs_1[0]
new_token = np.argmax(logits[0, - 1, :])
tokens.append(new_token)

num_layers = 24
kv_cache = {f'past_key_values.{i}.decoder.key': np.zeros((1, 2, max_sequence_length*2-1, 64), dtype=np.float32) for i in range(num_layers)}
kv_cache.update({f'past_key_values.{i}.decoder.value': np.zeros((1, 2, max_sequence_length*2-1, 64), dtype=np.float32) for i in range(num_layers)})

for layer in range(num_layers):
    kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,:start_len,:] = outputs_1[2*layer+1][:,:,:start_len,:]
    kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,:start_len,:] = outputs_1[2*layer+2][:,:,:start_len,:]


input_ids = np.zeros((1,1), dtype=inputs_1['input_ids'].dtype)
attention_mask = np.zeros((1,max_sequence_length*2), dtype=inputs_1['attention_mask'].dtype)
position_ids = np.array([start_len], dtype=np.int32)
input_ids[0][0] = new_token
attention_mask[0][:start_len+1] = 1
inputs_2 = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
inputs_2.update(kv_cache)

for i in range(max_sequence_length):
    outputs = session_2.run(None, inputs_2)

    logits = outputs[0]
    new_token = np.argmax(logits[0, - 1, :])
    
    tokens.append(new_token)

    inputs_2['position_ids'] += 1
    inputs_2['input_ids'][0][0] = new_token
    inputs_2["attention_mask"][0][start_len+1] = 1
    for layer in range(num_layers):
        kv_cache[f'past_key_values.{layer}.decoder.key'][:,:,start_len,:] = outputs[2*layer+1][:,:,-1,:]
        kv_cache[f'past_key_values.{layer}.decoder.value'][:,:,start_len,:] = outputs[2*layer+2][:,:,-1,:]

    inputs_2.update(kv_cache)
    start_len += 1

    if new_token == tokenizer.eos_token_id:
        break

print(tokenizer.decode(tokens, skip_special_tokens=False))