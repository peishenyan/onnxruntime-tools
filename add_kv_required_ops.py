import onnx
from onnx import helper
from onnx import TensorProto
import argparse
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('--cache', action='store_true', help='non kv cache or static kv cache', default=False)
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]    
model_name = model_prefix.split('/')[-1]
model_path = model_prefix+'.onnx'

model = onnx.load(model_path)

graph = model.graph
new_outputs = []

if args.cache:
    '''
    1. Cast: position_ids (INT32) -> position_ids_casted (INT64)
    2. Prepare indices for ScatterND: Concat(indices_left, Expand(position_ids_casted)), where indices_left is pre-defined 2-D indices prefix like [[0,0],[0,1],...[d_1-1,d_2-1]]
    3. Slice and Reshape: key_values.{layer}.decoder.key/value := present_key_values.{layer}.decoder.key/value[:,:,-1,:], Reshape(data, [d_1*d_2, d_4])
    4. ScatterND: data is past_key_values.{layer}.decoder.key/value, update data is key_values.{layer}.decoder.key/value
    '''

    indices = []
    kv_shape = None
    for tensor in graph.input:
        # 1. Cast position_ids to INT64
        if tensor.name == "position_ids":
            cast_node = helper.make_node(
                'Cast',
                inputs=[tensor.name],
                outputs=['position_ids_casted'],
                to=TensorProto.INT64
            )
            graph.node.extend([cast_node])
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    # Prepare initializers
    # For Slice
    starts_tensor = helper.make_tensor(
        name='starts_tensor',
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 0, kv_shape[2], 0], 
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
    
    # For Reshape
    d = kv_shape[0] * kv_shape[1]
    reshape_tensor_shape = helper.make_tensor(
        name='reshape_tensor_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[d, kv_shape[3]], 
    )

    # For indices_left
    dims_before = [kv_shape[0], kv_shape[1]]
    indices_left = np.indices(dims_before)
    indices_left_shape = [d, 2]
    flat_indices_left = np.stack(indices_left, axis=-1).reshape(-1, 2)
    indices_left_tensor = helper.make_tensor(
        name='scatter_indices_left',
        data_type=TensorProto.INT64,
        dims=indices_left_shape,
        vals=flat_indices_left
    )

    expand_shape_tensor = helper.make_tensor('position_ids_expand_shape', TensorProto.INT64, [2], [d, 1])
    graph.initializer.extend([starts_tensor, ends_tensor, axes_tensor, reshape_tensor_shape, indices_left_tensor, expand_shape_tensor])

    
    # 2. Prepare indices for ScatterND
    expand_node = helper.make_node(
        'Expand',
        name='position_ids_expand',
        inputs=['position_ids_casted', 'position_ids_expand_shape'],
        outputs=['expanded_position_ids'],
    )
    model.graph.node.extend([expand_node])

    concat_node = helper.make_node(
        'Concat',
        name='scatter_indices_concat',
        inputs=['scatter_indices_left', 'expanded_position_ids'],
        outputs=['scatter_indices'],
        axis=1
    )
    model.graph.node.extend([concat_node])

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
            outputs=['sliced_'+output_tensor.name]
        )
        graph.node.extend([slice_node])
        # 3.2 Reshape
        reshape_node = helper.make_node(
            'Reshape',
            name=output_tensor.name+'/Reshape',
            inputs=['sliced_'+output_tensor.name, 'reshape_tensor_shape'],
            outputs=['reshaped_'+output_tensor.name]
        )
        graph.node.extend([reshape_node])

        # Find the corresponding past_key_value_cache
        input_tensor = None
        for tensor in graph.input:
            if kv_name in tensor.name:
                input_tensor = tensor
        
        # 4. ScatterND
        scatter_node = helper.make_node(
            'ScatterND',
            name=output_tensor.name+'/ScatterND',
            inputs=[
                input_tensor.name,
                'scatter_indices',
                'reshaped_'+output_tensor.name
            ],
            outputs=['scatter_'+output_tensor.name]
        )
        graph.node.extend([scatter_node])

        shape_value = [kv_shape[0], kv_shape[1], kv_shape[2], kv_shape[3]]
        new_output =  helper.make_tensor_value_info('scatter_'+output_tensor.name, TensorProto.FLOAT, shape_value)
        new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])
    
else:
    indices = []
    kv_shape = None
    for tensor in graph.output:
        if kv_shape == None and 'key_values' in tensor.name:
            kv_shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    pads=[0, 0, 0, 0, 0, 0, kv_shape[2]-1, 0]
    pads_tensor = helper.make_tensor(
        name='pads_tensor',
        data_type=TensorProto.INT64,
        dims=[8],
        vals=pads,
    )
    graph.initializer.extend([pads_tensor])

    for i, output in enumerate(graph.output):
        if 'key_values' not in output.name:
            continue

        indices.append(i)
        pad_node = helper.make_node(
            'Pad',
            name=output.name+'/Pad',
            inputs=[output.name, 'pads_tensor'],
            outputs=['padded_'+output.name],
            mode='constant',
        )
        graph.node.extend([pad_node])
        shape_value = [kv_shape[0], kv_shape[1], kv_shape[2]*2-1, kv_shape[3]]
        new_output =  helper.make_tensor_value_info('padded_'+output.name, TensorProto.FLOAT, shape_value)
        new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])

new_model_path = model_prefix + '_padded.onnx'
convert_model_to_external_data(model, all_tensors_to_one_file=True, location=model_name+'_padded.onnx.data', size_threshold=1024, convert_attribute=False)
onnx.save(model, new_model_path)
print(f"Modified model saved to {new_model_path}")