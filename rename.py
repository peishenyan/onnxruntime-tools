import onnx
import argparse
from onnx.external_data_helper import convert_model_to_external_data

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_name', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-o', '--out_name', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-s', '--size', type=int, help="The Whisper model to convert", default=20000000)
args = parser.parse_args()
    
model_path = args.in_name
model_external_path = args.out_name
fn = args.out_name.split('/')[-1] + '.data'

onnx_model = onnx.load(model_path)
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location=fn, size_threshold=args.size)
onnx.save_model(onnx_model, model_external_path)