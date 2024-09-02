from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The model path to convert", default='logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx')
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]
model = model_prefix.split('/')[-1]

model_path = model_prefix+'.onnx'
model_quant_dynamic = model_prefix+'_INT8.onnx'

quantize_dynamic(model_path, model_quant_dynamic, weight_type=QuantType.QInt8, use_external_data_format=True)