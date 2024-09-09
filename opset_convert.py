import onnx
from onnx import version_converter
import argparse
from onnx.external_data_helper import convert_model_to_external_data

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-v', '--version', type=int, help="The opset version", default=21)
args = parser.parse_args()

model_prefix = args.model.split('.onnx')[0]
model = model_prefix.split('/')[-1]
# 加载ONNX模型
model_path = model_prefix+'.onnx'
original_model = onnx.load(model_path, load_external_data=False)

# 确定模型当前使用的opset版本
original_opset_version = original_model.opset_import[0].version
print(f"The opset version of the original model: {original_opset_version}")

# 目标opset版本
target_opset_version = args.version

# 尝试将模型转换到目标opset版本
try:
    # 使用ONNX的版本转换器进行转换
    converted_model = version_converter.convert_version(original_model, target_opset_version)
    print(f"Model has been successfully converted to opset version: {target_opset_version}")

    # 保存转换后的模型
    new_model_path = model_prefix+'_v'+str(target_opset_version)+'.onnx'
    onnx.save(converted_model, new_model_path)
    print(f"The converted model has been saved to the: {new_model_path}")

except Exception as e:
    print(f"Model conversion failure: {e}")


def update_weights_from_external_data(model_A_path, model_B_path, external_fn, external_data):
    onnx_model_B = onnx.load(model_B_path, load_external_data=True)
    onnx_model_A = onnx.load(model_A_path, load_external_data=False)
    
    inits_B = onnx_model_B.graph.initializer
    inits_A = onnx_model_A.graph.initializer
    for init_A, init_B  in zip(inits_A, inits_B):
        init_A.CopyFrom(init_B)
    
    convert_model_to_external_data(onnx_model_A, all_tensors_to_one_file=True, location=external_data, size_threshold=1024, convert_attribute=False)
    onnx.save_model(onnx_model_A, external_fn)


update_weights_from_external_data(new_model_path, model_path, new_model_path, model+'.onnx.data')