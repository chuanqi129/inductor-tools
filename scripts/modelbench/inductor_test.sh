# Not ready for updated benchmarks for nightly
export LD_PRELOAD=$LD_PRELOAD:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export USE_LLVM=False
# multiple / single / all
THREAD=${1:-all}
# first / last
CHANNELS=${2:-first}
LOG_DIR=${3:-inductor_log}
# all /torchbench /huggingface/ timm_models
MODEL_SUITE=${4:-all}
# multi-threads
multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [[ $CHANNELS == "first" ]]; then
        # channels first
        echo "Channels first testing...."
        if [ $MODEL_SUITE == "all" ]; then
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --output-dir=${LOG_DIR}/multi_threads_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        else
            python benchmarks/dynamo/runner.py --suites=${MODEL_SUITE} --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --output-dir=${LOG_DIR}/multi_threads_cf_${MODEL_SUITE}_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_${MODEL_SUITE}_bench_log_${timestamp}.log
        fi
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        if [ $MODEL_SUITE == "all" ]; then
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --channels-last --output-dir=${LOG_DIR}/multi_threads_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        else
            python benchmarks/dynamo/runner.py --suites=${MODEL_SUITE} --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --channels-last --output-dir=${LOG_DIR}/multi_threads_cl_${MODEL_SUITE}_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_${MODEL_SUITE}_bench_log_${timestamp}.log
        fi
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
        if [ $MODEL_SUITE == "all" ]; then
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --batch_size=1 --start_core 0 --end_core 0 --threads 1 --membind 0 --output-dir=${LOG_DIR}/single_thread_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
        else
            python benchmarks/dynamo/runner.py --suites=${MODEL_SUITE} --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --batch_size=1 --start_core 0 --end_core 0 --threads 1 --membind 0 --output-dir=${LOG_DIR}/single_thread_cf_${MODEL_SUITE}_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_${MODEL_SUITE}_bench_log_${timestamp}.log
        fi
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        if [ $MODEL_SUITE == "all" ]; then
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --channels-last --batch_size=1 --start_core 0 --end_core 0 --threads 1 --membind 0 --output-dir=${LOG_DIR}/single_thread_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
        else
            python benchmarks/dynamo/runner.py --suites=${MODEL_SUITE} --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=float32 --inference --compilers=inductor --channels-last --batch_size=1 --start_core 0 --end_core 0 --threads 1 --membind 0 --output-dir=${LOG_DIR}/single_thread_cl_${MODEL_SUITE}_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_${MODEL_SUITE}_bench_log_${timestamp}.log
        fi
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