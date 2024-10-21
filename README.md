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

## 1. From Transformers to ONNX
In `decoder-only.py`, we define a new "decoder-only" architecture model with existing "decoder-only" model imported from Huggingface/Transformers, called automodel. With different inputs, we have different forwarding process, which forms the automodel into two types of model: 1) the model with static shape at the first iteration, which has no kv cache as input and has `present_key_values` as output; 2) the model with static shape after the first iteration, which has `past_key_values` as input and `present_key_values` as output.
```
python decoder-only.py --static --export -m $model_name -l $max_sequence_length -c $max_cache_length
```

`$model_name` is the official name in Huggingface/Transformers like ``Qwen/Qwen2-0.5B-Instruct''.

The results are saved in `log/models/$model_name/${model_name}_decoder_1_prefill.onnx` and `log/models/$model_name/${model_name}_decoder_2_decode.onnx` files.  For example, the path is `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_1_prefill.onnx` and `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder__decoder_2_decode` if you export model `microsoft/Phi-3-mini-4k-instruct`.

## 2. Achieving ONNX model with external data
To avoid encountering the error `ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB:xxx`, you can convert and save the ONNX Model to External Data:
```
python convert.py -path $model_relative_path
```

`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_1_prefill.onnx`.

The results are saved in `log/models/$model_name/$model_path_ex.onnx` and `log/models/$model_name/$model_path_ex.onnx.data` files.

## 3. Achieving ONNX model with higher opset version
To avoid encountering the QDQ error, you can convert and save the ONNX Model to higher opset version:
```
python opset_convert.py -path $model_relative_path -v $opset_version
```

`$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_1_prefill_ex.onnx`. We recommend to use `$opset_version=21` for `Phi-3-mini-4k-instruct` model.

The results are saved in `log/models/$model_name/$model_path_v$opset_version.onnx` and `log/models/$model_name/$model_path_v$opset_version.onnx.data` files.


## 4. Quantize ONNX model to INT4 datatype
We can use `matmul_4bits_quantizer` in onnxruntime.quantization lib to quantize an ONNX model into INT4 datatype.
```
python quantize_int4.py -path $model_relative_path
```
If you want to fuse QDQ and matmul ops into MatMulNBits, please add params `-f`. `$model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_1_prefill_ex_v21.onnx`. 
The results are saved in `${model_relative_path}_INT4.onnx` files.

## 5. Align the output kv cache with input
To make it easy for running models with kv cache, we should align the output `present_key_values` for each model with input `past_key_values`.
```
python add_kv_required_ops_for_two.py -path $prefill_model_relative_path
python add_kv_required_ops_for_two.py -path $decode_model_relative_path --decode
```
`$prefill_model_relative_path` and `decode_model_relative_path` is the relative path of the model you want to convert, e.g., `logs/models/Phi_3_mini_4k_instruct/Phi_3_mini_4k_instruct_decoder_1_prefill_ex_v21_INT4_QDQ.onnx`.
The results are saved in `${model_relative_path}_padded.onnx` files.

## 6. Rename your model and make your model with both internal and external data
For easy usage in onnxruntime, rename your model and split the initializer of onnx model into both internal and external data:
```
python rename -i $input_model_path -o $output_model_paht -s $threshold_of_initializer
```

We recommend `$threshold_of_initializer=20000000` in default.

## 7. Run the ONNX model with Onnxruntime
```
python ort_two_pad.py
```

Please remind to modify the model path in `ort_two_pad.py`.