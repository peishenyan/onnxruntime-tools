# onnxruntime-tools

Some tools to help us to work around dirty work.

## Requirements
```
accelerate==0.33.0
huggingface_hub==0.24.5
numpy==1.26.2
onnxruntime==1.18.1
optimum==1.22.0.dev0
torch==2.1.2
transformers==4.42.4
```
**WARNING!!!** If you want to run the Microsoft/Phi-3-mini-4k-instruct model, you need to build and install the latest onnxruntime lib from github souce code, otherwise you will encounter the error `onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Trilu(14) node with name '/decoder/Trilu' · Issue #16189 · microsoft/onnxruntime (github.com)`.

## From Transformers to ONNX

```
python decoder-only.py --static --export -m $model_name
```

`$model_name` is the official name in Huggingface/Transformers like ``Qwen/Qwen2-0.5B-Instruct''.

The results are saved in `log/models/$model_name/$model_name.onnx` files.

## Quantize ONNX model to INT8 datatype
```
python quantize.py -m $model_name 
```

The results are saved in `log/models/$model_name/$model_name_INT8.onnx` files.

## External data for ONNX model
If you encounter the error `ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB:xxx`, you can convert and save the ONNX Model to External Data:
```
python convert.py -m $model_name 
```

The results are saved in `log/models/$model_name/$model_name_ex.onnx` files.

## Run the ONNX model with Onnxruntime
```
python ort.py -m $model_name [-q: use quantized model]
```
microsoft/Phi-3-mini-4k-instruct model requires some Huggingface rights, you should input the tokens if needed.