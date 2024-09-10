import onnx
import argparse
from onnx.external_data_helper import convert_model_to_external_data

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
args = parser.parse_args()
model_prefix = args.model.split('.onnx')[0]
model = model_prefix.split('/')[-1]
    
model_path = model_prefix+'.onnx'
model_external_path = model_prefix+'_ex.onnx'

fn = model+'_ex.onnx.data'

onnx_model = onnx.load(model_path)
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location=fn, size_threshold=1024, convert_attribute=False)
onnx.save_model(onnx_model, model_external_path)