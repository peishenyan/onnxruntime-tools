python decoder-only-dynamic.py -m microsoft/Phi-3-mini-4k-instruct
echo 'LOG: Export dynamic model finished!!!'
model_path="logs/models/dynamic/Phi_3_mini_4k_instruct/"
temp_path="logs/models/dynamic/done/"
model_name="Phi_3_mini_4k_instruct_decoder_static_cache"

python convert.py -path ${model_path}${model_name}.onnx
echo 'LOG 1: Generating model with external data finished.'
python opset_convert.py -path ${model_path}${model_name}_ex.onnx -v 21
echo 'LOG 1: Generating model with new opset finished.'
rm ${model_path}${model_name}_ex.onnx*
python quantize_int4.py -path ${model_path}${model_name}_ex_v21.onnx
echo 'LOG 1: Quantizing model finished.'
rm ${model_path}${model_name}_ex_v21.onnx*

mkdir ${temp_path}
mv ${model_path}${model_name}_ex_v21_INT4_QDQ.onnx* ${temp_path}
rm -r ${model_path}