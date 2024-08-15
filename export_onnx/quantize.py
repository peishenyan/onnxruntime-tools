from onnxruntime.quantization import quantize_dynamic, QuantType

model = 'Phi_3_mini_4k_instruct' # 'Qwen2_0.5B_Instruct' 'TinyLlama_1.1B_Chat_v1.0' 'hi_3_mini_4k_instruct'
model_path_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm.onnx'
model_path_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm.onnx'
model_quant_dynamic_1 = 'logs/models/'+ model + '/' + model +'_decoder_static_non_kvcache_lm_INT8.onnx'
model_quant_dynamic_2 = 'logs/models/'+ model + '/' + model +'_decoder_static_kvcache_128_lm_INT8.onnx'

quantize_dynamic(model_path_1, model_quant_dynamic_1, weight_type=QuantType.QInt8, use_external_data_format=True)
quantize_dynamic(model_path_2, model_quant_dynamic_2, weight_type=QuantType.QInt8, use_external_data_format=True)