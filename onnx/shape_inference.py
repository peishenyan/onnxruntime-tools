import onnx
from onnx import shape_inference

name = './models/qwen/Qwen2_0.5B_Instruct_decoder_dynamic_non_kvcache_lm'
# Load the ONNX model
model = onnx.load(name+'.onnx')
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, name+'_infer.onnx')
onnx.checker.check_model(name+'_infer.onnx')