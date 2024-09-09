# onnxruntime-tools

Some tools to export, convert and run onnx models. Now only support some decoder-only LLM models.

## Supported model list
```
Qwen/Qwen2-0.5B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
microsoft/Phi-3-mini-4k-instruct
```

## Requirements
```
accelerate==0.33.0
huggingface_hub==0.24.5
numpy==1.26.2
onnx==1.16.2
onnxruntime==1.19.0
optimum==1.22.0.dev0
torch==2.1.2
transformers==4.42.4
```
> **WARNING!!!** If you want to run the Microsoft/Phi-3-mini-4k-instruct model, you need to build and install the latest onnxruntime lib from github souce code, otherwise you will encounter the error `onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Trilu(14) node with name '/decoder/Trilu' · Issue #16189 · microsoft/onnxruntime (github.com)`.
>
> At the same time, microsoft/Phi-3-mini-4k-instruct model requires some Huggingface rights, you should input the tokens if needed.

## From Transformers to ONNX
In `decoder-only.py`, we define a new "decoder-only" architecture model with existing "decoder-only" model imported from Huggingface/Transformers, called automodel. With different inputs, we have different forwarding process, which forms the automodel into two types of model: 1) the model with static shape at the first iteration, which has no kv cache as input and has `present_key_values` as output; 2) the model with static shape after the first iteration, which has `past_key_values` as input and `present_key_values` as output.
```
python decoder-only.py --static --export -m $model_name -l $max_sequence_length
```

`$model_name` is the official name in Huggingface/Transformers like ``Qwen/Qwen2-0.5B-Instruct''.

The results are saved in `log/models/$model_name/${model_name}_decoder_dynamic_non_kvcache_lm.onnx` and `log/models/$model_name/${model_name}_decoder_dynamic_kvcache_${max_sequence_length}_lm.onnx` files.  For example, the path is `logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx` if you export `Qwen/Qwen2-0.5B-Instruct` with max sequence length 128.

## Achieving ONNX model with external data
To avoid encountering the error `ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB:xxx`, you can convert and save the ONNX Model to External Data:
```
python convert.py -path $model_relative_path
```

`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx`.

The results are saved in `log/models/$model_name/$model_path_ex.onnx` and `log/models/$model_name/$model_path_ex.onnx.data` files.

## Achieving ONNX model with higher opset version
To avoid encountering the QDQ error, you can convert and save the ONNX Model to higher opset version:
```
python opset_convert.py -path $model_relative_path -v $opset_version
```

`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm_ex.onnx`.

The results are saved in `log/models/$model_name/$model_path_v$opset_version.onnx` and `log/models/$model_name/$model_path_v$opset_version.onnx.data` files.

## Quantize ONNX model to INT8 datatype
We can use `quantize_dynamic` in onnxruntime.quantization lib to quantize an ONNX model into INT8 datatype.
```
python quantize.py -path $model_relative_path
```
`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx`.
The results are saved in `${model_relative_path}_INT8.onnx` files.

## Quantize ONNX model to INT4 datatype
We can use `matmul_4bits_quantizer` in onnxruntime.quantization lib to quantize an ONNX model into INT4 datatype.
```
python quantize_int4.py -path $model_relative_path [-f: fuse QDQ+Matmul into MatmulNBits]
```
`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Qwen2_0.5B_Instruct/Qwen2_0.5B_Instruct_decoder_static_kvcache_128_lm.onnx`.
The results are saved in `${model_relative_path}_INT4.onnx` files.

## Run the ONNX model with Onnxruntime
```
python ort.py -m $model_name [-q 0/4/8: (do not) use quantized INT4 or INT8 model]
```