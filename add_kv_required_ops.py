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
    indices = []
    # Cast position_ids to INT64
    for tensor in graph.input:
        if tensor.name == "position_ids":
            tensor
            cast_node = helper.make_node(
                'Cast',
                inputs=[tensor.name],  # 输入张量名称
                outputs=['position_ids_casted'],  # 输出张量名称
                to=TensorProto.INT64  # 目标数据类型
            )

            graph.node.extend([cast_node])

    # For each kv_cache do slice and scatterND
    for i, output_tensor in enumerate(graph.output):
        if 'key_values' not in output_tensor.name:
            continue
        
        # Slice present_key_value_cache
        starts_tensor = helper.make_tensor(
            name=output_tensor.name+'.starts_tensor',
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[0, 0, output_tensor.type.tensor_type.shape.dim[2].dim_value-1, 0], 
        )
        ends_tensor = helper.make_tensor(
            name=output_tensor.name+'.ends_tensor',
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[output_tensor.type.tensor_type.shape.dim[0].dim_value, output_tensor.type.tensor_type.shape.dim[1].dim_value, output_tensor.type.tensor_type.shape.dim[2].dim_value, output_tensor.type.tensor_type.shape.dim[3].dim_value], 
        )
        axes_tensor = helper.make_tensor(
            name=output_tensor.name+'.axes_tensor',
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[0, 1, 2, 3], 
        )
        graph.initializer.extend([starts_tensor, ends_tensor, axes_tensor])

        indices.append(i)
        kv_name = output_tensor.name.split('sent_')[-1]
        # print(kv_name)
        slice_node = helper.make_node(
            'Slice',
            name=output_tensor.name+'/Slice_0',
            inputs=[output_tensor.name, output_tensor.name+'.starts_tensor', output_tensor.name+'.ends_tensor', output_tensor.name+'.axes_tensor'],
            outputs=['sliced_0_'+output_tensor.name]
        )
        graph.node.extend([slice_node])

        d = output_tensor.type.tensor_type.shape.dim[0].dim_value * output_tensor.type.tensor_type.shape.dim[1].dim_value
        slice_tensor_shape = helper.make_tensor(
            name=output_tensor.name+'.slice_tensor_shape',
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[d, output_tensor.type.tensor_type.shape.dim[3].dim_value], 
        )
        graph.initializer.extend([slice_tensor_shape])

        reshape_node = helper.make_node(
            'Reshape',
            name=output_tensor.name+'/Slice',
            inputs=['sliced_0_'+output_tensor.name, output_tensor.name+'.slice_tensor_shape'],
            outputs=['sliced_'+output_tensor.name]
        )
        graph.node.extend([reshape_node])

        # Find the corresponding past_key_value_cache
        input_tensor = None
        for tensor in graph.input:
            if kv_name in tensor.name:
                input_tensor = tensor

        # Prepare indices for ScatterND
        
        dims_before = [output_tensor.type.tensor_type.shape.dim[0].dim_value, output_tensor.type.tensor_type.shape.dim[1].dim_value]
        indices_left = np.indices(dims_before)
        indices_left_shape = [d, 2]
        flat_indices_left = np.stack(indices_left, axis=-1).reshape(-1, 2)

        indices_left_tensor = helper.make_tensor(
            output_tensor.name+'/scatter_indices_left',
            TensorProto.INT64,
            indices_left_shape,
            flat_indices_left
        )
        graph.initializer.extend([indices_left_tensor])

        shape_tensor = helper.make_tensor(output_tensor.name+'/position_ids_expand_shape', TensorProto.INT64, [2], [d, 1])  # 扩展到[d, 1]
        model.graph.initializer.extend([shape_tensor])

        expand_node = helper.make_node(
            'Expand',
            name=output_tensor.name+'/position_ids_expand',
            inputs=['position_ids_casted', output_tensor.name+'/position_ids_expand_shape'],
            outputs=[output_tensor.name+'/expanded_position_ids'],
        )
        model.graph.node.extend([expand_node])
        
        concat_node = helper.make_node(
            'Concat',
            name=output_tensor.name+'/scatter_indices_oncat',
            inputs=[output_tensor.name+'/scatter_indices_left', output_tensor.name+'/expanded_position_ids'],
            outputs=[output_tensor.name+'/scatter_indices'],
            axis=1
        )
        model.graph.node.extend([concat_node])
        
        scatter_node = helper.make_node(
            'ScatterND',
            name=output_tensor.name+'/ScatterND',
            inputs=[
                input_tensor.name,  # 输入张量
                output_tensor.name+'/scatter_indices',  # 索引张量
                'sliced_'+output_tensor.name  # 切片张量作为更新值
            ],
            outputs=['scatter_'+output_tensor.name],  # Scatter操作的输出张量名称
        )
        graph.node.extend([scatter_node])

        shape_value = [output_tensor.type.tensor_type.shape.dim[0].dim_value, output_tensor.type.tensor_type.shape.dim[1].dim_value, output_tensor.type.tensor_type.shape.dim[2].dim_value-1, output_tensor.type.tensor_type.shape.dim[3].dim_value]
        new_output =  helper.make_tensor_value_info('scatter_'+output_tensor.name, TensorProto.FLOAT, shape_value)
        new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])
    
else:
    indices = []

    for i, output in enumerate(graph.output):
        if 'key_values' not in output.name:
            continue

        pads=[0, 0, 0, 0, 0, 0, output.type.tensor_type.shape.dim[2].dim_value-1, 0]
        pads_tensor = helper.make_tensor(
            name=output.name+'.pads_tensor',
            data_type=TensorProto.INT64,
            dims=[8],  # 假设我们需要在两个维度上进行Padding
            vals=pads,  # Pad值，这里是一个示例
        )
        graph.initializer.extend([pads_tensor])

        indices.append(i)
        pad_node = helper.make_node(
            'Pad',
            name=output.name+'/Pad',
            inputs=[output.name, output.name+'.pads_tensor'],
            outputs=['padded_'+output.name],
            mode='constant',
        )
        graph.node.extend([pad_node])
        shape_value = [output.type.tensor_type.shape.dim[0].dim_value, output.type.tensor_type.shape.dim[1].dim_value, 2*output.type.tensor_type.shape.dim[2].dim_value-1, output.type.tensor_type.shape.dim[3].dim_value]
        new_output =  helper.make_tensor_value_info('padded_'+output.name, TensorProto.FLOAT, shape_value)
        new_outputs.append(new_output)

    for i, j in enumerate(indices):
        graph.output.pop(j)
        graph.output.insert(j, new_outputs[i])


new_model_path = model_prefix + '_padded.onnx'
convert_model_to_external_data(model, all_tensors_to_one_file=True, location=model_name+'_padded.onnx.data', size_threshold=1024, convert_attribute=False)
onnx.save(model, new_model_path)
print(f"Modified model saved to {new_model_path}")