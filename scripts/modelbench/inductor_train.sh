export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

# multiple / single / all
THREAD=${1:-multiple}
# first / last
CHANNELS=${2:-first}
LOG_DIR=${3:-inductor_log_train}
mkdir -p $LOG_DIR


torchbench="mobilenet_v2,doctr_reco_predictor"
huggingface="MobileBertForQuestionAnswering,Speech2Text2ForCausalLM,BlenderbotSmallForCausalLM,MobileBertForMaskedLM"
timm="fbnetv3_b,mobilenetv3_large_100,mnasnet_100,lcnet_050"

torchbench_model_list=($(echo "${torchbench}" |sed 's/,/ /g'))
huggingface_model_list=($(echo "${huggingface}" |sed 's/,/ /g'))
timm_model_list=($(echo "${timm}" |sed 's/,/ /g'))


# multi-threads
multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [[ $CHANNELS == "first" ]]; then
        # channels first
        echo "Channels first testing...."
        for torchbench_model in ${torchbench_model_list[@]}
        do
            # Commands for torchbench for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/torchbench.py --performance --float32 -dcpu  --training --inductor   --no-skip --dashboard --only ${torchbench_model} --cold_start_latency --output=${LOG_DIR}/multi_threads_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
            # Commands for torchbench for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu --training --inductor   --no-skip --dashboard --only ${torchbench_model} --output=${LOG_DIR}/multi_threads_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log             
        done

        for huggingface_model in ${huggingface_model_list[@]}
        do
            # Commands for huggingface for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --training --inductor   --no-skip --dashboard --only ${huggingface_model} --cold_start_latency --output=${LOG_DIR}/multi_threads_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
            # Commands for huggingface for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu  --training --inductor   --no-skip --dashboard --only ${huggingface_model}  --output=${LOG_DIR}/multi_threads_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        done

        for timm_model in ${timm_model_list[@]}
        do
            # Commands for timm_models for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --training --inductor   --no-skip --dashboard --only ${timm_model} --cold_start_latency --output=${LOG_DIR}/multi_threads_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
            # Commands for timm_models for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --training --inductor   --no-skip --dashboard --only ${timm_model} --output=${LOG_DIR}/multi_threads_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        done       
        
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Skip Channels last testing...."
    else
        echo "Please check channels foramt with first / last."
    fi
}

# single-thread
single_thread_test() {
    export OMP_NUM_THREADS=1
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [[ $CHANNELS == "first" ]]; then
        # channels first
        echo "Channels first testing...."
        for torchbench_model in ${torchbench_model_list[@]}
        do
            # Commands for torchbench for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --training --inductor   --no-skip --dashboard  --cold_start_latency --batch_size 1 --threads 1 --only ${torchbench_model} --output=${LOG_DIR}/single_thread_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
            # Commands for torchbench for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu  --training --inductor   --no-skip --dashboard --batch_size 1 --threads 1 --only ${torchbench_model} --output-directory=${LOG_DIR}/single_thread_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
        done

        for huggingface_model in ${huggingface_model_list[@]}
        do
            # Commands for huggingface for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --training --inductor   --no-skip --dashboard --cold_start_latency --batch_size 1 --threads 1 --only ${huggingface_model}  --output=${LOG_DIR}/single_thread_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
            # Commands for huggingface for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --training --inductor   --no-skip --dashboard --batch_size 1 --threads 1 --only ${huggingface_model} --output=${LOG_DIR}/single_thread_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
        done

        for timm_model in ${timm_model_list[@]}
        do
            # Commands for timm_models for device=cpu, dtype=float32 for training and for performance testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --training --inductor   --no-skip --dashboard --cold_start_latency --batch_size 1 --threads 1 --only ${timm_model} --output=${LOG_DIR}/single_thread_cf_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
            # Commands for timm_models for device=cpu, dtype=float32 for training and for accuracy testing
            python -m torch.backends.xeon.run_cpu --core_list 0 --ncores_per_instance 1 benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --training --inductor   --no-skip --dashboard --batch_size 1 --threads 1 --only ${timm_model} --output=${LOG_DIR}/single_thread_cf_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
        done
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Skip Channels last testing...."
    else
        echo "Please check channels foramt with first / last."
    fi
}

if [[ $THREAD == "multiple" ]]; then
    echo "multi-threads testing...."
    multi_threads_test
elif [[ $THREAD == "single" ]]; then
    echo "single-thread testing...."
    single_thread_test
elif [[ $THREAD == "all" ]]; then
    echo "1. multi-threads testing...."
    multi_threads_test
    echo "2. single-thread testing...."
    single_thread_test
else
    echo "Please check thread mode with multiple / single / all"
fi
