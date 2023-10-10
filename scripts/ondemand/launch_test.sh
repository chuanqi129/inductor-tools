DT=${1:-float32} # float32 / amp

if [[ $DT == "float32" ]]; then
    echo "================================Benchmarking FP32 TORCHBENCH Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance torchbench float32 first static default 0 2>&1 | tee torchbench_fp32_infer_test.log 

    echo "================================Benchmarking FP32 HF Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance huggingface float32 first static default 0 2>&1 | tee hf_fp32_infer_test.log 

    echo "================================Benchmarking FP32 timm_models Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance timm_models float32 first static default 0 2>&1 | tee timm_fp32_infer_test.log
else
    echo "================================Benchmarking amp TORCHBENCH Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance torchbench amp first static default 0 2>&1 | tee torchbench_amp_infer_test.log 

    echo "================================Benchmarking amp HF Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance huggingface amp first static default 0 2>&1 | tee hf_amp_infer_test.log 

    echo "================================Benchmarking amp timm_models Inference================================================"
    TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" bash vec_inductor_test.sh multiple inference performance timm_models amp first static default 0 2>&1 | tee timm_amp_infer_test.log    
fi