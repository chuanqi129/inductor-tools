# if you want to get the --token-latency breakdown, please follow the bkc
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# git checkout v4.31.0
# git apply gpu-models/LLM/profile_patch
# python setup.py install

# the PoC weekly check:
# beam=1, bs=1, input=1024, out=128
# beam=4, bs=1, input=1024, out=128
beam=4
bs=1
input=128
out=128
iter=10

## distributed setting
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh

# GPT-J-6b
Run_benchmark_gpt-j-6b() {
    model=EleutherAI/gpt-j-6B
    sub_model_name=gpt-j-6B
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 4 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16 2>&1 | tee log_e2e_ds_4
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16 2>&1 | tee log_e2e_ds_2
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16 2>&1 | tee log_e2e_ds_single
    mv log_e2e_ds ${dir}
#    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
#    mv profile*pt ${dir}
#    mv trace.json ${dir}
}


## Llama-7b
Run_benchmark_llama-7b() {
    model=decapoda-research/llama-7b-hf
    sub_model_name=llama-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama-13b
Run_benchmark_llama-13b() {
    model=decapoda-research/llama-13b-hf
    sub_model_name=llama-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-7b
Run_benchmark_llama2-7b() {
    model=/mllnvme0/llama2-7b
    sub_model_name=llama2-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-13b
Run_benchmark_llama2-13b() {
    model=/mllnvme0/llama2-13b
    sub_model_name=llama2-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}



## Llama2-34b
Run_benchmark_llama2-34b() {
    model=codellama/CodeLlama-34b-hf
    sub_model_name=llama2-34b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-70b
Run_benchmark_llama2-70b() {
    model=/mllnvme0/llama2
    sub_model_name=llama2-70b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum4
    mkdir -p ${dir}
    mpirun -np 4 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 4 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## OPT-6.7b
Run_benchmark_opt-6.7b() {
    model=facebook/opt-6.7b
    sub_model_name=opt-6.7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## OPT-30b
Run_benchmark_opt-30b() {
    model=facebook/opt-30b
    sub_model_name=opt-30b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## BLOOM-7b
Run_benchmark_bloom-7b() {
    model=bigscience/bloom-7b1
    sub_model_name=bloom-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}
    mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## BLOOM-176b
Run_benchmark_bloom-176b() {
    model=bigscience/bloom
    sub_model_name=bloom-176b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum8
    mkdir -p ${dir}
    mpirun -np 8 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e_ds
    mv log_e2e_ds ${dir}
    PROFILE=1 mpirun -np 8 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

# For a detailed explanation on how to run Mixtral-8x7b, please refer to mixtral8x7b.md
Run_benchmark_mixtral8x7b() {
    # sharding huggingface weights firstly
    model_id="mistralai/Mixtral-8x7B-v0.1"
    # change this path to your local path if necessary
    model=~/mixtral/shard/
    python create_shard_model.py -m $model_id --save-path  ${model}  --dtype bfloat16
    sub_model_name=mixtral
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}_ranknum2
    mkdir -p ${dir}    
    # run model
    mpirun -np 2 --prepend-rank \
        python -u run_generation_with_deepspeed.py \
            --benchmark -m ${model} \
            --sub-model-name ${sub_model_name} \
            --num-beams ${beam} --num-iter ${iter} \
            --batch-size ${bs} --input-tokens ${input} \
            --max-new-tokens ${out} --device xpu \
            --ipex \
            --dtype float16  | tee log_e2e_ds
}


main() {
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
    export ENABLE_SDP_FUSION=1

#    export HF_HOME=/mllnvme0/huggingface/
#    export HF_DATASETS_OFFLINE=1
#    export TRANSFORMERS_OFFLINE=1
#    export HF_EVALUATE_OFFLINE=1
    # export TORCH_CCL_DISABLE_ALLREDUCE=1

    Run_benchmark_gpt-j-6b
#    Run_benchmark_llama-7b
#    Run_benchmark_llama-13b
#    Run_benchmark_llama2-7b
#    Run_benchmark_llama2-13b
#    Run_benchmark_llama2-34b
#    Run_benchmark_llama2-70b
#    Run_benchmark_opt-6.7b
#    Run_benchmark_opt-30b
#    Run_benchmark_bloom-7b
#    Run_benchmark_bloom-176b
    # Comment this out as this takes much time to shard data and run
    # Run_benchmark_mixtral8x7b
}

main
