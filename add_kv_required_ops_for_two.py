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
    '''
    1. Cast: position_ids (INT32) -> position_ids_casted (INT64)
    2. Prepare indices for ScatterND: Concat(indices_left, Expand(position_ids_casted)), where indices_left is pre-defined 2-D indices prefix like [[0,0],[0,1],...[d_1-1,d_2-1]]
    3. Slice and Reshape: key_values.{layer}.decoder.key/value := present_key_values.{layer}.decoder.key/value[:,:,-1,:], Reshape(data, [d_1*d_2, d_4])
    4. ScatterND: data is past_key_values.{layer}.decoder.key/value, update data is key_values.{layer}.decoder.key/value
    '''

    indices = []
    kv_shape = None
    for tensor in graph.input:
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    # Prepare initializers
    # For Slice
    starts_tensor = helper.make_tensor(
        name='starts_tensor',
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 0, 1, 0], 
    )
    ends_tensor = helper.make_tensor(
        name='ends_tensor',
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[kv_shape[0], kv_shape[1], kv_shape[2]+1, kv_shape[3]], 
    )
    axes_tensor = helper.make_tensor(
        name='axes_tensor',
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 1, 2, 3], 
    )
    
    graph.initializer.extend([starts_tensor, ends_tensor, axes_tensor])

    for i, output_tensor in enumerate(graph.output):
        if 'key_values' not in output_tensor.name:
            continue
        
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
    for tensor in graph.input:
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]


    range_tensor = helper.make_tensor('gather_range', onnx.TensorProto.INT64, [kv_shape[2]], range(kv_shape[2]))
    offset_tensor = helper.make_tensor('gather_offset', onnx.TensorProto.INT64, [], [kv_shape[2]-1])
    graph.initializer.extend([range_tensor, offset_tensor])

    argmax_node = helper.make_node(
        'ArgMax',
        name='Gather_position',
        inputs=['attention_mask'],
        outputs=['argmax_res'],
        axis=1,
        keepdims=0,
        select_last_index=1,
    )

    sub_node = helper.make_node(
        'Sub',
        inputs=['argmax_res', 'gather_offset'],
        outputs=['gather_addition']
    )

    add_node = helper.make_node(
        'Add',
        inputs=['gather_range', 'gather_addition'],
        outputs=['gather_indices']
    )

    graph.node.extend([argmax_node, sub_node, add_node])

    for i, output in enumerate(graph.output):
        if 'key_values' not in output.name:
            continue

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

new_model_path = model_prefix + '_padded.onnx'
convert_model_to_external_data(model, all_tensors_to_one_file=True, location=model_name+'_padded.onnx.data', size_threshold=20000000, convert_attribute=False)
onnx.save(model, new_model_path)
print(f"Modified model saved to {new_model_path}")