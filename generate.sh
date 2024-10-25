python decoder-only.py --static --export -m microsoft/Phi-3-mini-4k-instruct -l 128 -c 256
echo 'LOG: Export prefill model finished!!!'
python decoder-only.py --static --export -m microsoft/Phi-3-mini-4k-instruct -l 128 -c 256 --decode
echo 'LOG: Export decode model finished!!!'
model_path="logs/models/Phi_3_mini_4k_instruct/"
temp_path="logs/models/done/"
model_list=("Phi_3_mini_4k_instruct_decoder_1_prefill" "Phi_3_mini_4k_instruct_decoder_2_decode")

python convert.py -path ${model_path}${model_list[0]}.onnx
echo 'LOG 1: Generating model with external data finished.'
python opset_convert.py -path ${model_path}${model_list[0]}_ex.onnx -v 21
echo 'LOG 1: Generating model with new opset finished.'
rm ${model_path}${model_list[0]}_ex.onnx*
python quantize_int4.py -path ${model_path}${model_list[0]}_ex_v21.onnx
echo 'LOG 1: Quantizing model finished.'
rm ${model_path}${model_list[0]}_ex_v21.onnx*
python pad_two.py -path ${model_path}${model_list[0]}_ex_v21_INT4_QDQ.onnx
echo 'LOG 1: Aligning model output with input finished.'
rm ${model_path}${model_list[0]}_ex_v21_INT4_QDQ.onnx*
python rename.py -i ${model_path}${model_list[0]}_ex_v21_INT4_QDQ_final.onnx -o ${model_path}1_prefill_INT4_final.onnx -s 20000000
rm ${model_path}${model_list[0]}_ex_v21_INT4_QDQ_final.onnx*
echo 'LOG 1: Renaming model finished.'

python convert.py -path ${model_path}${model_list[1]}.onnx
echo 'LOG 2: Generating model with external data finished.'
python opset_convert.py -path ${model_path}${model_list[1]}_ex.onnx -v 21
echo 'LOG 2: Generating model with new opset finished.'
rm ${model_path}${model_list[1]}_ex.onnx*
python quantize_int4.py -path ${model_path}${model_list[1]}_ex_v21.onnx
echo 'LOG 2: Quantizing model finished.'
rm ${model_path}${model_list[1]}_ex_v21.onnx*
python pad_two.py -path ${model_path}${model_list[1]}_ex_v21_INT4_QDQ.onnx --decode
echo 'LOG 2: Aligning model output with input finished.'
rm ${model_path}${model_list[1]}_ex_v21_INT4_QDQ.onnx*
python rename.py -i ${model_path}${model_list[1]}_ex_v21_INT4_QDQ_final.onnx -o ${model_path}2_decode_INT4_final.onnx -s 20000000  
rm ${model_path}${model_list[1]}_ex_v21_INT4_QDQ_final.onnx*
echo 'LOG 2: Renaming model finished.'

mkdir ${temp_path}
mv ${model_path}1_prefill_INT4_final.onnx* ${temp_path}
mv ${model_path}2_decode_INT4_final.onnx* ${temp_path}
rm -r ${model_path}