echo -e "========================================================================="
echo -e "huggingface accuracy results check"
echo -e "========================================================================="

cd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface
cd amp_bf16
echo -e "============ Acc Check for HF amp_bf16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_bf16=csv_lines_inf-1
num_passed_amp_bf16_inf=$(grep "pass" inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_inf=num_total_amp_bf16-num_passed_amp_bf16_inf
amp_bf16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_inf'/'$num_total_amp_bf16')*100}'`
echo "num_total_amp_bf16: $num_total_amp_bf16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_bf16_inf: $num_passed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_inf: $num_failed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_inf_acc_pass_rate: $amp_bf16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_bf16_tra=$(grep "pass" inductor_huggingface_amp_bf16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_tra=num_total_amp_bf16-num_passed_amp_bf16_tra
amp_bf16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_tra'/'$num_total_amp_bf16')*100}'`
echo "num_passed_amp_bf16_tra: $num_passed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_tra: $num_failed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_tra_acc_pass_rate: $amp_bf16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../amp_fp16
echo -e "============ Acc Check for HF amp_fp16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_fp16=csv_lines_inf-1
num_passed_amp_fp16_inf=$(grep "pass" inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_inf=num_total_amp_fp16-num_passed_amp_fp16_inf
amp_fp16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_inf'/'$num_total_amp_fp16')*100}'`
echo "num_total_amp_fp16: $num_total_amp_fp16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_fp16_inf: $num_passed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_inf: $num_failed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_inf_acc_pass_rate: $amp_fp16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_fp16_tra=$(grep "pass" inductor_huggingface_amp_fp16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_tra=num_total_amp_fp16-num_passed_amp_fp16_tra
amp_fp16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_tra'/'$num_total_amp_fp16')*100}'`
echo "num_passed_amp_fp16_tra: $num_passed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_tra: $num_failed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_tra_acc_pass_rate: $amp_fp16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../bfloat16
echo -e "============ Acc Check for HF bfloat16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_huggingface_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_total_bfloat16=csv_lines_inf-1
num_passed_bfloat16_inf=$(grep "pass" inductor_huggingface_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_inf=num_total_bfloat16-num_passed_bfloat16_inf
bfloat16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_inf'/'$num_total_bfloat16')*100}'`
echo "num_total_bfloat16: $num_total_bfloat16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_bfloat16_inf: $num_passed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_inf: $num_failed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_inf_acc_pass_rate: $bfloat16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_bfloat16_tra=$(grep "pass" inductor_huggingface_bfloat16_training_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_tra=num_total_bfloat16-num_passed_bfloat16_tra
bfloat16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_tra'/'$num_total_bfloat16')*100}'`
echo "num_passed_bfloat16_tra: $num_passed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_tra: $num_failed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_tra_acc_pass_rate: $bfloat16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float16
echo -e "============ Acc Check for HF float16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_huggingface_float16_inference_xpu_accuracy.csv | wc -l)
let num_total_float16=csv_lines_inf-1
num_passed_float16_inf=$(grep "pass" inductor_huggingface_float16_inference_xpu_accuracy.csv | wc -l)
let num_failed_float16_inf=num_total_float16-num_passed_float16_inf
float16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_inf'/'$num_total_float16')*100}'`
echo "num_total_float16: $num_total_float16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float16_inf: $num_passed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_inf: $num_failed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_inf_acc_pass_rate: $float16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float16_tra=$(grep "pass" inductor_huggingface_float16_training_xpu_accuracy.csv | wc -l)
let num_failed_float16_tra=num_total_float16-num_passed_float16_tra
float16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_tra'/'$num_total_float16')*100}'`
echo "num_passed_float16_tra: $num_passed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_tra: $num_failed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_tra_acc_pass_rate: $float16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float32
echo -e "============ Acc Check for HF float32 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_huggingface_float32_inference_xpu_accuracy.csv | wc -l)
let num_total_float32=csv_lines_inf-1
num_passed_float32_inf=$(grep "pass" inductor_huggingface_float32_inference_xpu_accuracy.csv | wc -l)
let num_failed_float32_inf=num_total_float32-num_passed_float32_inf
float32_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_inf'/'$num_total_float32')*100}'`
echo "num_total_float32: $num_total_float32" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float32_inf: $num_passed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_inf: $num_failed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_inf_acc_pass_rate: $float32_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float32_tra=$(grep "pass" inductor_huggingface_float32_training_xpu_accuracy.csv | wc -l)
let num_failed_float32_tra=num_total_float32-nunum_passed_float32_tra
float32_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_tra'/'$num_total_float32')*100}'`
echo "num_passed_float32_tra: $num_passed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_tra: $num_failed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_tra_acc_pass_rate: $float32_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

echo -e "========================================================================="
echo -e "timm_models accuracy results check"
echo -e "========================================================================="

cd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/timm_models
cd amp_bf16
echo -e "============ Acc Check for TM amp_bf16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_timm_models_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_bf16=csv_lines_inf-1
num_passed_amp_bf16_inf=$(grep "pass" inductor_timm_models_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_inf=num_total_amp_bf16-num_passed_amp_bf16_inf
amp_bf16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_inf'/'$num_total_amp_bf16')*100}'`
echo "num_total_amp_bf16: $num_total_amp_bf16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_bf16_inf: $num_passed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_inf: $num_failed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_inf_acc_pass_rate: $amp_bf16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_bf16_tra=$(grep "pass" inductor_timm_models_amp_bf16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_tra=num_total_amp_bf16-num_passed_amp_bf16_tra
amp_bf16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_tra'/'$num_total_amp_bf16')*100}'`
echo "num_passed_amp_bf16_tra: $num_passed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_tra: $num_failed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_tra_acc_pass_rate: $amp_bf16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../amp_fp16
echo -e "============ Acc Check for TM amp_fp16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_timm_models_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_fp16=csv_lines_inf-1
num_passed_amp_fp16_inf=$(grep "pass" inductor_timm_models_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_inf=num_total_amp_fp16-num_passed_amp_fp16_inf
amp_fp16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_inf'/'$num_total_amp_fp16')*100}'`
echo "num_total_amp_fp16: $num_total_amp_fp16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_fp16_inf: $num_passed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_inf: $num_failed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_inf_acc_pass_rate: $amp_fp16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_fp16_tra=$(grep "pass" inductor_timm_models_amp_fp16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_tra=num_total_amp_fp16-num_passed_amp_fp16_tra
amp_fp16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_tra'/'$num_total_amp_fp16')*100}'`
echo "num_passed_amp_fp16_tra: $num_passed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_tra: $num_failed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_tra_acc_pass_rate: $amp_fp16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../bfloat16
echo -e "============ Acc Check for TM bfloat16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_timm_models_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_total_bfloat16=csv_lines_inf-1
num_passed_bfloat16_inf=$(grep "pass" inductor_timm_models_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_inf=num_total_bfloat16-num_passed_bfloat16_inf
bfloat16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_inf'/'$num_total_bfloat16')*100}'`
echo "num_total_bfloat16: $num_total_bfloat16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_bfloat16_inf: $num_passed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_inf: $num_failed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_inf_acc_pass_rate: $bfloat16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_bfloat16_tra=$(grep "pass" inductor_timm_models_bfloat16_training_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_tra=num_total_bfloat16-num_passed_bfloat16_tra
bfloat16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_tra'/'$num_total_bfloat16')*100}'`
echo "num_passed_bfloat16_tra: $num_passed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_tra: $num_failed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_tra_acc_pass_rate: $bfloat16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float16
echo -e "============ Acc Check for TM float16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_timm_models_float16_inference_xpu_accuracy.csv | wc -l)
let num_total_float16=csv_lines_inf-1
num_passed_float16_inf=$(grep "pass" inductor_timm_models_float16_inference_xpu_accuracy.csv | wc -l)
let num_failed_float16_inf=num_total_float16-num_passed_float16_inf
float16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_inf'/'$num_total_float16')*100}'`
echo "num_total_float16: $num_total_float16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float16_inf: $num_passed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_inf: $num_failed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_inf_acc_pass_rate: $float16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float16_tra=$(grep "pass" inductor_timm_models_float16_training_xpu_accuracy.csv | wc -l)
let num_failed_float16_tra=num_total_float16-num_passed_float16_tra
float16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_tra'/'$num_total_float16')*100}'`
echo "num_passed_float16_tra: $num_passed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_tra: $num_failed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_tra_acc_pass_rate: $float16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float32
echo -e "============ Acc Check for TM float32 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_timm_models_float32_inference_xpu_accuracy.csv | wc -l)
let num_total_float32=csv_lines_inf-1
num_passed_float32_inf=$(grep "pass" inductor_timm_models_float32_inference_xpu_accuracy.csv | wc -l)
let num_failed_float32_inf=num_total_float32-num_passed_float32_inf
float32_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_inf'/'$num_total_float32')*100}'`
echo "num_total_float32: $num_total_float32" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float32_inf: $num_passed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_inf: $num_failed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_inf_acc_pass_rate: $float32_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float32_tra=$(grep "pass" inductor_timm_models_float32_training_xpu_accuracy.csv | wc -l)
let num_failed_float32_tra=num_total_float32-nunum_passed_float32_tra
float32_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_tra'/'$num_total_float32')*100}'`
echo "num_passed_float32_tra: $num_passed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_tra: $num_failed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_tra_acc_pass_rate: $float32_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

echo -e "========================================================================="
echo -e "torchbench accuracy results check"
echo -e "========================================================================="

cd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/torchbench
cd amp_bf16
echo -e "============ Acc Check for torchbench amp_bf16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_torchbench_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_bf16=csv_lines_inf-1
num_passed_amp_bf16_inf=$(grep "pass" inductor_torchbench_amp_bf16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_inf=num_total_amp_bf16-num_passed_amp_bf16_inf
amp_bf16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_inf'/'$num_total_amp_bf16')*100}'`
echo "num_total_amp_bf16: $num_total_amp_bf16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_bf16_inf: $num_passed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_inf: $num_failed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_inf_acc_pass_rate: $amp_bf16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_bf16_tra=$(grep "pass" inductor_torchbench_amp_bf16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_bf16_tra=num_total_amp_bf16-num_passed_amp_bf16_tra
amp_bf16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_tra'/'$num_total_amp_bf16')*100}'`
echo "num_passed_amp_bf16_tra: $num_passed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_bf16_tra: $num_failed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_bf16_tra_acc_pass_rate: $amp_bf16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../amp_fp16
echo -e "============ Acc Check for torchbench amp_fp16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
csv_lines_inf=$(cat inductor_torchbench_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_total_amp_fp16=csv_lines_inf-1
num_passed_amp_fp16_inf=$(grep "pass" inductor_torchbench_amp_fp16_inference_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_inf=num_total_amp_fp16-num_passed_amp_fp16_inf
amp_fp16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_inf'/'$num_total_amp_fp16')*100}'`
echo "num_total_amp_fp16: $num_total_amp_fp16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_amp_fp16_inf: $num_passed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_inf: $num_failed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_inf_acc_pass_rate: $amp_fp16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_amp_fp16_tra=$(grep "pass" inductor_torchbench_amp_fp16_training_xpu_accuracy.csv | wc -l)
let num_failed_amp_fp16_tra=num_total_amp_fp16-num_passed_amp_fp16_tra
amp_fp16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_tra'/'$num_total_amp_fp16')*100}'`
echo "num_passed_amp_fp16_tra: $num_passed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_amp_fp16_tra: $num_failed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "amp_fp16_tra_acc_pass_rate: $amp_fp16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../bfloat16
echo -e "============ Acc Check for torchbench bfloat16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_torchbench_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_total_bfloat16=csv_lines_inf-1
num_passed_bfloat16_inf=$(grep "pass" inductor_torchbench_bfloat16_inference_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_inf=num_total_bfloat16-num_passed_bfloat16_inf
bfloat16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_inf'/'$num_total_bfloat16')*100}'`
echo "num_total_bfloat16: $num_total_bfloat16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_bfloat16_inf: $num_passed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_inf: $num_failed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_inf_acc_pass_rate: $bfloat16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_bfloat16_tra=$(grep "pass" inductor_torchbench_bfloat16_training_xpu_accuracy.csv | wc -l)
let num_failed_bfloat16_tra=num_total_bfloat16-num_passed_bfloat16_tra
bfloat16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_tra'/'$num_total_bfloat16')*100}'`
echo "num_passed_bfloat16_tra: $num_passed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_bfloat16_tra: $num_failed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "bfloat16_tra_acc_pass_rate: $bfloat16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float16
echo -e "============ Acc Check for torchbench float16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_torchbench_float16_inference_xpu_accuracy.csv | wc -l)
let num_total_float16=csv_lines_inf-1
num_passed_float16_inf=$(grep "pass" inductor_torchbench_float16_inference_xpu_accuracy.csv | wc -l)
let num_failed_float16_inf=num_total_float16-num_passed_float16_inf
float16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_inf'/'$num_total_float16')*100}'`
echo "num_total_float16: $num_total_float16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float16_inf: $num_passed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_inf: $num_failed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_inf_acc_pass_rate: $float16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float16_tra=$(grep "pass" inductor_torchbench_float16_training_xpu_accuracy.csv | wc -l)
let num_failed_float16_tra=num_total_float16-num_passed_float16_tra
float16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_tra'/'$num_total_float16')*100}'`
echo "num_passed_float16_tra: $num_passed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float16_tra: $num_failed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float16_tra_acc_pass_rate: $float16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

cd ../float32
echo -e "============ Acc Check for torchbench float32 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
csv_lines_inf=$(cat inductor_torchbench_float32_inference_xpu_accuracy.csv | wc -l)
let num_total_float32=csv_lines_inf-1
num_passed_float32_inf=$(grep "pass" inductor_torchbench_float32_inference_xpu_accuracy.csv | wc -l)
let num_failed_float32_inf=num_total_float32-num_passed_float32_inf
float32_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_inf'/'$num_total_float32')*100}'`
echo "num_total_float32: $num_total_float32" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_passed_float32_inf: $num_passed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_inf: $num_failed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_inf_acc_pass_rate: $float32_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
num_passed_float32_tra=$(grep "pass" inductor_torchbench_float32_training_xpu_accuracy.csv | wc -l)
let num_failed_float32_tra=num_total_float32-nunum_passed_float32_tra
float32_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_tra'/'$num_total_float32')*100}'`
echo "num_passed_float32_tra: $num_passed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "num_failed_float32_tra: $num_failed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
echo "float32_tra_acc_pass_rate: $float32_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log