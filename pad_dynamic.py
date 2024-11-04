import onnx
from onnx import helper
from onnx import TensorProto
import argparse
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-sink', '--attn_sink', type=int, help='The length of attention sink tokens', default=4)
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]    
model_name = model_prefix.split('/')[-1]
model_path = model_prefix+'.onnx'

model = onnx.load(model_path)
graph = model.graph

attention_sink_len = args.attn_sink
new_input = helper.make_tensor_value_info('attention_sink_len', onnx.TensorProto.INT32, [1])
model.graph.input.append(new_input)

new_outputs = []
indices = []

kv_shape = None
kv_denote = None
for tensor in graph.input:
    if kv_shape == None and 'key_values' in tensor.name:
        kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        kv_denote = [dim.dim_param for dim in tensor.type.tensor_type.shape.dim]

zero_tensor = helper.make_tensor('value_zero', onnx.TensorProto.INT32, [], [0])
one_tensor = helper.make_tensor('value_one', onnx.TensorProto.INT64, [], [1])
# sink_1d_tensor = helper.make_tensor('sink_len_1D', onnx.TensorProto.INT32, [1], [attention_sink_len])
# sink_len_tensor = helper.make_tensor('sink_len', onnx.TensorProto.INT32, [], [attention_sink_len])
# sink_len_int64_tensor = helper.make_tensor('sink_len_int64', onnx.TensorProto.INT64, [], [attention_sink_len])
# sink_range_tensor = helper.make_tensor('sink_range', onnx.TensorProto.INT32, [attention_sink_len], range(attention_sink_len))
graph.initializer.extend([zero_tensor, one_tensor]) #, sink_1d_tensor, sink_len_tensor, sink_len_int64_tensor, sink_range_tensor])


cast_node = helper.make_node(
    'Cast',
    name='attention_sink_len_cast',
    inputs=['attention_sink_len'],
    outputs=['sink_len_int64'],
    to=TensorProto.INT64
)
 # Get the value of max_cache_length and max_text_length
shape_node_1 = helper.make_node(
    'Shape',
    name='attn_mask_shape',
    inputs=['attention_mask'],
    outputs=['attention_mask_shape'],
    start=-1
) # max_cache_length+max_text_length

shape_node_2 = helper.make_node(
    'Shape',
    name='input_shape',
    inputs=['input_ids'],
    outputs=['input_ids_shape'],
    start=-1
) # max_text_length

max_cache_len_node = helper.make_node(
    'Sub',
    name='max_cache_len_sub',
    inputs=['attention_mask_shape', 'input_ids_shape'],
    outputs=['max_cache_len']
) # max_cache_length

# To implement attention sink cache, we should construct left gather indices for sink cache and right gather indices for the rest cache.
right_cache_len_node = helper.make_node(
    'Sub',
    name='right_cache_len_sub',
    inputs=['max_cache_len', 'sink_len_int64'],
    outputs=['right_cache_len']
)
graph.node.extend([cast_node, shape_node_1, shape_node_2, max_cache_len_node, right_cache_len_node])

# Prepare range(max_cache_len), range(max_seq_len), and range(max_seq_len+max_cache_len)
ones_node_1 = helper.make_node(
    'ConstantOfShape',
    name='one_of_attn_mask',
    inputs=['attention_mask_shape'],
    outputs=['attention_mask_ones'],
    value=helper.make_tensor('value_1', onnx.TensorProto.INT32, [1], [1])
)
ones_node_2 = helper.make_node(
    'ConstantOfShape',
    name='one_of_max_cache_len',
    inputs=['max_cache_len'],
    outputs=['max_cache_len_ones'],
    value=helper.make_tensor('value_1', onnx.TensorProto.INT32, [1], [1])
)
slice_node = helper.make_node(
    'Slice',
    name='one_of_right_cache_len',
    inputs=['max_cache_len_ones', 'attention_sink_len', 'mask_gather_addition'],
    outputs=['right_cache_len_ones']
)
ones_node_3 = helper.make_node(
    'ConstantOfShape',
    name='one_of_input_ids_shape',
    inputs=['input_ids_shape'],
    outputs=['input_ids_shape_ones'],
    value=helper.make_tensor('value_1', onnx.TensorProto.INT32, [1], [1])
)
ones_node_4 = helper.make_node(
    'ConstantOfShape',
    name='one_attention_sink_shape',
    inputs=['sink_len_int64'],
    outputs=['attention_sink_len_ones'],
    value=helper.make_tensor('value_1', onnx.TensorProto.INT32, [1], [1])
)
range_node_1 = helper.make_node(
    'CumSum',
    name='range_of_attn_mask',
    inputs=['attention_mask_ones', 'value_zero'],
    outputs=['mask_range'],
    reverse=1,
)
range_node_2 = helper.make_node(
    'CumSum',
    name='range_of_right_cache_len',
    inputs=['right_cache_len_ones', 'value_zero'],
    outputs=['right_gather_range'],
)
range_node_3 = helper.make_node(
    'CumSum',
    name='range_of_input_ids_shape',
    inputs=['input_ids_shape_ones', 'value_zero'],
    outputs=['mask_gather_range'],
    exclusive=1,
) # start from 0
range_node_4 = helper.make_node(
    'CumSum',
    name='range_of_attention_sink_len',
    inputs=['attention_sink_len_ones', 'value_zero'],
    outputs=['sink_range'],
    exclusive=1,
) # start from 0
graph.node.extend([ ones_node_1, ones_node_2, slice_node, ones_node_3, ones_node_4, range_node_1, range_node_2, range_node_3, range_node_4])


### Slice attention mask
cast_node = helper.make_node(
    'Cast',
    name='max_cache_lenn_cast',
    inputs=['max_cache_len'],
    outputs=['mask_gather_addition'],
    to=TensorProto.INT32
)

add_node = helper.make_node(
    'Add',
    name='mask_gather_range_add',
    inputs=['mask_gather_range', 'mask_gather_addition'],
    outputs=['mask_gather_indices']
)

gather_node = helper.make_node(
    'Gather',
    name='attention_mask_gather',
    inputs=['attention_mask', 'mask_gather_indices'],
    outputs=['sliced_attention_mask'],
    axis=1
)
graph.node.extend([cast_node, add_node, gather_node])

# Calculate argmax for right attention mask and argmin for all attention mask 
mul_node_1 = helper.make_node(
    'Mul',
    name='attention_mask_mul',
    inputs=['attention_mask', 'mask_range'],
    outputs=['attention_mask_pos']
)

mul_node_2 = helper.make_node(
    'Mul',
    name='sliced_attention_mask_mul',
    inputs=['sliced_attention_mask', 'mask_gather_indices'],
    outputs=['right_attention_mask_pos']
)

argmax_node = helper.make_node(
    'ArgMax',
    name='Gather_end',
    inputs=['right_attention_mask_pos'],
    outputs=['argmax_res'],
    axis=1,
    keepdims=0
)

argmin_node = helper.make_node(
    'ArgMax',
    name='Gather_start',
    inputs=['attention_mask_pos'],
    outputs=['min_one_pos'],
    axis=1,
    keepdims=0
)
graph.node.extend([mul_node_1, mul_node_2, argmax_node, argmin_node])

cast_node_1 = helper.make_node(
    'Cast',
    name='gather_addition_cast',
    inputs=['argmax_res'],
    outputs=['right_gather_addition'],
    to=TensorProto.INT32
) # INT32 argmax_res

add_node = helper.make_node(
    'Add',
    name='add',
    inputs=['argmax_res', 'value_one'],
    outputs=['input_len']
) # argmax_res + 1 = input_len

min_node = helper.make_node(
    'Min',
    name='min_for_left_gather',
    inputs=['input_len', 'min_one_pos'],
    outputs=['left_gather_addition_0']
) # left gather indices start at min(input_len, max(0,max_cache_len-start_len)) = min(input_len, min_one_pos)

cast_node_2 = helper.make_node(
    'Cast',
    name='left_gather_addition_cast',
    inputs=['left_gather_addition_0'],
    outputs=['left_gather_addition'],
    to=TensorProto.INT32
)

add_node_0 = helper.make_node(
    'Add',
    name='add_0',
    inputs=['sink_range', 'left_gather_addition'],
    outputs=['left_gather_indices']
)

add_node_1 = helper.make_node(
    'Add',
    name='add_1',
    inputs=['right_gather_range', 'right_gather_addition'],
    outputs=['right_gather_indices_0']
)

add_node_2 = helper.make_node(
    'Add',
    name='add_2',
    inputs=['right_gather_indices_0', 'attention_sink_len'],
    outputs=['right_gather_indices']
)

concat_node = helper.make_node(
    'Concat',
    name='gather_indices_concat',
    inputs=['left_gather_indices', 'right_gather_indices'],
    outputs=['gather_indices'],
    axis=0
)
graph.node.extend([cast_node_1, min_node, cast_node_2, add_node, add_node_0, add_node_1, add_node_2, concat_node])

for i, output in enumerate(graph.output):
    if 'logits' in output.name:
        indices.append(i)

        gather_node = helper.make_node(
            'Gather',
            name='logits_gather',
            inputs=[output.name, 'right_gather_addition'],
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
            name='gather/'+output.name,
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