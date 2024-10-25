import onnx
from onnx import helper
from onnx import TensorProto
import argparse
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]    
model_name = model_prefix.split('/')[-1]
model_path = model_prefix+'.onnx'

model = onnx.load(model_path)

graph = model.graph
new_outputs = []


indices = []

kv_shape = None
kv_denote = None
for tensor in graph.input:
    if kv_shape == None and 'key_values' in tensor.name:
        kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        kv_denote = [dim.dim_param for dim in tensor.type.tensor_type.shape.dim]

zero_tensor = helper.make_tensor('value_zero', onnx.TensorProto.INT32, [], [0])
graph.initializer.extend([zero_tensor])

# Get the value of max_cache_length and max_text_length
shape_node_1 = helper.make_node(
    'Shape',
    name='attn_mask_shape',
    inputs=['attention_mask'],
    outputs=['attention_mask_shape'],
    start=-1
)

shape_node_2 = helper.make_node(
    'Shape',
    name='input_shape',
    inputs=['input_ids'],
    outputs=['input_ids_shape'],
    start=-1
)

max_cache_len_node = helper.make_node(
    'Sub',
    name='max_cache_len_sub',
    inputs=['attention_mask_shape', 'input_ids_shape'],
    outputs=['max_cache_len']
)

ones_node_1 = helper.make_node(
    'ConstantOfShape',
    name='one_of_attn_mask',
    inputs=['attention_mask_shape'],
    outputs=['attention_mask_ones'],
    value=helper.make_tensor('value_one', onnx.TensorProto.INT32, [1], [1])
)

ones_node_2 = helper.make_node(
    'ConstantOfShape',
    name='one_of_max_cache_len',
    inputs=['max_cache_len'],
    outputs=['max_cache_len_ones'],
    value=helper.make_tensor('value_one', onnx.TensorProto.INT32, [1], [1])
)

range_node_1 = helper.make_node(
    'CumSum',
    name='range_of_attn_mask',
    inputs=['attention_mask_ones', 'value_zero'],
    outputs=['mask_range'],
)

# [1,2,3,...]
range_node_2 = helper.make_node(
    'CumSum',
    name='range_of_max_cache_len',
    inputs=['max_cache_len_ones', 'value_zero'],
    outputs=['gather_range'],
)
graph.node.extend([shape_node_1, shape_node_2, max_cache_len_node, ones_node_1, ones_node_2, range_node_1, range_node_2])


mul_node = helper.make_node(
    'Mul',
    inputs=['attention_mask', 'mask_range'],
    outputs=['attention_mask_pos']
)

# input_ids = [1], attention_mask_pos = 128
argmax_node = helper.make_node(
    'ArgMax',
    name='Gather_position',
    inputs=['attention_mask_pos'],
    outputs=['argmax_res'],
    axis=1,
    keepdims=0
)

sub_node = helper.make_node(
    'Sub',
    name='gather_sub',
    inputs=['argmax_res', 'max_cache_len'],
    outputs=['gather_addition_0']
)

cast_node = helper.make_node(
    'Cast',
    name='gather_addition_cast',
    inputs=['gather_addition_0'],
    outputs=['gather_addition'],
    to=TensorProto.INT32
)

add_node = helper.make_node(
    'Add',
    inputs=['gather_range', 'gather_addition'],
    outputs=['gather_indices']
)

graph.node.extend([mul_node, argmax_node, sub_node, cast_node, add_node])

for i, output in enumerate(graph.output):
    if 'logits' in output.name:
        indices.append(i)

        gather_node = helper.make_node(
            'Gather',
            name='logits_gather',
            inputs=[output.name, 'gather_addition'],
            outputs=['sliced_'+output.name],
            axis=1
        )
        logits_argmax_node = helper.make_node(
            'ArgMax',
            name='output_token',
            inputs=['sliced_'+output.name],
            outputs=['token_id_0'],
            axis=2,
            keepdims=0
        )
        output_cast_node = helper.make_node(
            'Cast',
            name='output_token_cast',
            inputs=['token_id_0'],
            outputs=['token_id_1'],
            to=TensorProto.INT32
        )
        flatten_node = helper.make_node(
            'Flatten',
            name='output_token_flatten',
            inputs=['token_id_1'],
            outputs=['token_id']
        )
        graph.node.extend([gather_node, logits_argmax_node, output_cast_node, flatten_node])
        shape_value = [1,1]
        new_output =  helper.make_tensor_value_info('token_id', TensorProto.INT32, shape_value)
        new_outputs.append(new_output)


    if 'key_values' in output.name:
        indices.append(i)

        gather_node = helper.make_node(
            'Gather',
            inputs=[output.name, 'gather_indices'],
            outputs=['new_'+output.name],
            axis=2
        )
        graph.node.extend([gather_node])
        
        shape_value = [kv_shape[0], kv_shape[1], kv_denote[2], kv_shape[3]]
        new_output =  helper.make_tensor_value_info('new_'+output.name, TensorProto.FLOAT, shape_value)
        new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])

new_model_path = model_prefix + '_final.onnx'
convert_model_to_external_data(model, all_tensors_to_one_file=True, location=model_name+'_final.onnx.data', size_threshold=20000000, convert_attribute=False)
onnx.save(model, new_model_path)
print(f"Modified model saved to {new_model_path}")