from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-ex', '--external', action='store_true', help='use external data model', default=False)
args = parser.parse_args()
try:
    model = args.model.split('/')[1].replace('-', '_')
except:
    model = args.model.replace('-', '_')


# model = 'Phi_3_mini_4k_instruct' # 'Qwen2_0.5B_Instruct' 'TinyLlama_1.1B_Chat_v1.0' 'hi_3_mini_4k_instruct'
if args.external:
    ex = '_ex'
else:
    ex = ''

model_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm'+ex+'.onnx'
model_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm'+ex+'.onnx'

model_quant_dynamic_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm_INT8.onnx'
model_quant_dynamic_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm_INT8.onnx'

quantize_dynamic(model_path_1, model_quant_dynamic_1, weight_type=QuantType.QInt8, use_external_data_format=True)
quantize_dynamic(model_path_2, model_quant_dynamic_2, weight_type=QuantType.QInt8, use_external_data_format=True)