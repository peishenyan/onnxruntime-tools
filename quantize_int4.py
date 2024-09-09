from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
    quantize
)
from pathlib import Path
import onnx
import argparse
from onnx import shape_inference

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--model', type=str, help="The Whisper model to convert", default='Qwen/Qwen2-0.5B-Instruct')
parser.add_argument('-f', '--fuse', action='store_true', help='fuse QDQ and matmul into MatmulNBits', default=False)
args = parser.parse_args()

model_prefix = args.model.split('.onnx')[0]
model = model_prefix.split('/')[-1]

if args.fuse:
    quant_format = quant_utils.QuantFormat.QOperator
    q_format = ''
else:
    quant_format = quant_utils.QuantFormat.QDQ
    q_format = '_QDQ'

# logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_static_non_kvcache_lm.onnx logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_static_kvcache_128_lm.onnx
model_fp32_path = model_prefix+'.onnx'
model_int4_path = model_prefix+'_INT4'+q_format+'.onnx'

quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
  block_size=128, # 2's exponential and >= 16
  is_symmetric=True, # if true, quantize to Int4. otherwsie, quantize to uint4.
  accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
  quant_format=quant_format
)

model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
  model, 
  nodes_to_exclude=None, # specify a list of nodes to exclude from quantizaiton
  # nodes_to_include=None, # specify a list of nodes to force include from quantization
  algo_config=quant_config,)
quant.process()
quant.model.save_model_to_file(
  model_int4_path,
  True) # save data to external file