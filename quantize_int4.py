from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
    quantize
)
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-q', '--quantize', action='store_true', help='use quantized INT8 model', default=False)
parser.add_argument('-ex', '--external', action='store_true', help='use external data model', default=False)
parser.add_argument('-f', '--fuse', action='store_true', help='fuse QDQ and matmul into MatmulNBits', default=False)
args = parser.parse_args()
try:
    model = args.model.split('/')[1].replace('-', '_')
except:
    model = args.model.replace('-', '_')

if args.fuse:
    quant_format = quant_utils.QuantFormat.QOperator
else:
    quant_format = quant_utils.QuantFormat.QDQ

if args.external:
    ex = '_ex'
else:
    ex = ''

# logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_static_non_kvcache_lm.onnx logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_static_kvcache_128_lm.onnx
model_fp32_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm'+ex+'.onnx'
model_int4_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm_INT4.onnx'
quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
  block_size=128, # 2's exponential and >= 16
  is_symmetric=True, # if true, quantize to Int4. otherwsie, quantize to uint4.
  accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
  quant_format=quant_format
)

model_1 = quant_utils.load_model_with_shape_infer(Path(model_fp32_path_1))
quant_1 = matmul_4bits_quantizer.MatMul4BitsQuantizer(
  model_1, 
  nodes_to_exclude=None, # specify a list of nodes to exclude from quantizaiton
  # nodes_to_include=None, # specify a list of nodes to force include from quantization
  algo_config=quant_config,)
quant_1.process()
quant_1.model.save_model_to_file(
  model_int4_path_1,
  True) # save data to external file

model_fp32_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm'+ex+'.onnx'
model_int4_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm_INT4.onnx'
model_2 = quant_utils.load_model_with_shape_infer(Path(model_fp32_path_1))
quant_2 = matmul_4bits_quantizer.MatMul4BitsQuantizer(
  model_2, 
  nodes_to_exclude=None, # specify a list of nodes to exclude from quantizaiton
  # nodes_to_include=None, # specify a list of nodes to force include from quantization
  algo_config=quant_config,)
quant_2.process()
quant_2.model.save_model_to_file(
  model_int4_path_2,
  True) # save data to external file