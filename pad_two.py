import onnx
from onnx import helper
from onnx import TensorProto
import argparse
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('--decode', action='store_true', help='whether the model is for decode or prefill', default=False)
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]    
model_name = model_prefix.split('/')[-1]
model_path = model_prefix+'.onnx'

model = onnx.load(model_path)

graph = model.graph
new_outputs = []

if args.decode:
    indices = []
    kv_shape = None
    for tensor in graph.input:
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    # Prepare initializers
    # For Slice
    starts_tensor = helper.make_tensor(
        name='starts_tensor',
        data_type=TensorProto.INT32,
        dims=[4],
        vals=[0, 0, 1, 0], 
    )
    ends_tensor = helper.make_tensor(
        name='ends_tensor',
        data_type=TensorProto.INT32,
        dims=[4],
        vals=[kv_shape[0], kv_shape[1], kv_shape[2]+1, kv_shape[3]], 
    )
    axes_tensor = helper.make_tensor(
        name='axes_tensor',
        data_type=TensorProto.INT32,
        dims=[4],
        vals=[0, 1, 2, 3], 
    )
    
    graph.initializer.extend([starts_tensor, ends_tensor, axes_tensor])

    for i, output_tensor in enumerate(graph.output):
        if 'logits' in output_tensor.name:
            indices.append(i)
            logits_argmax_node = helper.make_node(
                'ArgMax',
                name='output_token',
                inputs=['logits'],
                outputs=['token_id_0'],
                axis=2,
                keepdims=0
            )
            cast_node = helper.make_node(
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
            graph.node.extend([logits_argmax_node, cast_node, flatten_node])
            shape_value = [1,1]
            new_output =  helper.make_tensor_value_info('token_id', TensorProto.INT32, shape_value)
            new_outputs.append(new_output)

        if 'key_values' in output_tensor.name:        
            # 3.1 Slice present_key_value_cache
            indices.append(i)
            kv_name = output_tensor.name.split('sent_')[-1]
            slice_node = helper.make_node(
                'Slice',
                name=output_tensor.name+'/Slice',
                inputs=[output_tensor.name, 'starts_tensor', 'ends_tensor', 'axes_tensor'],
                outputs=['new_'+output_tensor.name]
            )
            graph.node.extend([slice_node])

            shape_value = [kv_shape[0], kv_shape[1], kv_shape[2], kv_shape[3]]
            new_output =  helper.make_tensor_value_info('new_'+output_tensor.name, TensorProto.FLOAT, shape_value)
            new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])
    
else:
    indices = []
    kv_shape = None
    mask_shape = None
    for tensor in graph.input:
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        if 'attention_mask' in tensor.name:
            mask_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    range_tensor = helper.make_tensor('gather_range', onnx.TensorProto.INT64, [kv_shape[2]], range(kv_shape[2]))
    mask_range_tensor = helper.make_tensor('mask_range', onnx.TensorProto.INT32, [mask_shape[1]], range(mask_shape[1]))
    offset_tensor = helper.make_tensor('gather_offset', onnx.TensorProto.INT64, [], [kv_shape[2]-1])
    logits_offset_tensor = helper.make_tensor('logits_offset', onnx.TensorProto.INT64, [], [kv_shape[2]])
    graph.initializer.extend([range_tensor, mask_range_tensor, offset_tensor, logits_offset_tensor])

    mul_node = helper.make_node(
        'Mul',
        inputs=['attention_mask', 'mask_range'],
        outputs=['attention_mask_pos']
    )

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
        inputs=['argmax_res', 'gather_offset'],
        outputs=['gather_addition']
    )

    add_node = helper.make_node(
        'Add',
        inputs=['gather_range', 'gather_addition'],
        outputs=['gather_indices']
    )

    sub_node_2 = helper.make_node(
        'Sub',
        name='logits_sub',
        inputs=['argmax_res', 'logits_offset'],
        outputs=['logits_index']
    )
    graph.node.extend([mul_node, argmax_node, sub_node, add_node, sub_node_2])

    for i, output in enumerate(graph.output):
        if 'logits' in output.name:
            indices.append(i)

            gather_node = helper.make_node(
                'Gather',
                name='logits_gather',
                inputs=[output.name, 'logits_index'],
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
            cast_node = helper.make_node(
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
            graph.node.extend([gather_node, logits_argmax_node, cast_node, flatten_node])
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
            
            shape_value = [kv_shape[0], kv_shape[1], kv_shape[2], kv_shape[3]]
            new_output =  helper.make_tensor_value_info('new_'+output.name, TensorProto.FLOAT, shape_value)
            new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])

new_model_path = model_prefix + '_final.onnx'
convert_model_to_external_data(model, all_tensors_to_one_file=True, location=model_name+'_final.onnx.data', size_threshold=20000000, convert_attribute=False)
onnx.save(model, new_model_path)
print(f"Modified model saved to {new_model_path}")