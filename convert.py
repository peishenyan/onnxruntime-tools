import onnx
import argparse
from onnx.external_data_helper import convert_model_to_external_data

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-ex', '--external', action='store_true', help='use external data model', default=False)
args = parser.parse_args()
try:
    model = args.model.split('/')[1].replace('-', '_')
except:
    model = args.model.replace('-', '_')
    
model_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm.onnx'
model_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm.onnx'

model_external_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm_ex.onnx'
model_external_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm_ex.onnx'

fn_1 = model +'_decoder_static_non_kvcache_lm_ex.onnx_data'
fn_2 = model +'_decoder_static_kvcache_128_lm_ex.onnx_data'

onnx_model_1 = onnx.load(model_path_1)
convert_model_to_external_data(onnx_model_1, all_tensors_to_one_file=True, location=fn_1, size_threshold=1024, convert_attribute=False)
onnx.save_model(onnx_model_1, model_external_path_1)


onnx_model_2 = onnx.load(model_path_2)
convert_model_to_external_data(onnx_model_2, all_tensors_to_one_file=True, location=fn_2, size_threshold=1024, convert_attribute=False)
onnx.save_model(onnx_model_2, model_external_path_2)