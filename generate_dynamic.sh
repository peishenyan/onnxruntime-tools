python decoder-only-dynamic.py -m microsoft/Phi-3-mini-4k-instruct
echo 'LOG: Export dynamic model finished!!!'
model_path="logs/models/dynamic_1/Phi_3_mini_4k_instruct/"
temp_path="logs/models/dynamic_1/done/"
model_name="Phi_3_mini_4k_instruct_decoder_static_cache"

python convert.py -path ${model_path}${model_name}.onnx
echo 'LOG: Generating model with external data finished.'
python opset_convert.py -path ${model_path}${model_name}_ex.onnx -v 21
echo 'LOG: Generating model with new opset finished.'
rm ${model_path}${model_name}_ex.onnx*
python quantize_int4.py -path ${model_path}${model_name}_ex_v21.onnx
echo 'LOG: Quantizing model finished.'
rm ${model_path}${model_name}_ex_v21.onnx*
python pad_dynamic.py -path ${model_path}${model_name}_ex_v21_INT4_QDQ.onnx
echo 'LOG: Padding model finished.'
rm ${model_path}${model_name}_ex_v21_INT4_QDQ.onnx*

mkdir ${temp_path}
mv ${model_path}${model_name}_ex_v21_INT4_QDQ_final.onnx* ${temp_path}
rm -r ${model_path}
python rename.py -i ${temp_path}${model_name}_ex_v21_INT4_QDQ_final.onnx -o ${temp_path}Phi_3_mini_4k_instruct_decoder_static_cache_MatmulNbits.onnx
rm ${temp_path}${model_name}_ex_v21_INT4_QDQ_final.onnx*