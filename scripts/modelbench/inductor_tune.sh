export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# multiple / single / all
SUITE=${1:-torchbench}
# first / last
CHANNELS=${2:-first}
LOG_DIR=${3:-dynamo_ww48_5}

log_path=$LOG_DIR/${SUITE}_tune
mkdir -p ${log_pah}

# multi-threads
multi_threads_tune() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    timestamp=$(date +%Y%m%d_%H%M%S)
    BS_ALL="1 2 4 8 16 32 64 128 256 512 1024"
    for BS in ${BS_ALL}; do
        echo BS:${BS}
        if [[ $CHANNELS == "first" ]]; then
            # channels first
            echo "Channels first testing...."
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${log_path}/archive --devices=cpu --dtypes=float32 --inference --suite=${SUITE} --compilers=inductor --batch_size=${BS} --output-dir=${log_path}/multi_threads_cf_logs_${timestamp} 2>&1 | tee ${log_path}/multi_threads_${SUITE}_model_bench_bs_${BS}_log_${timestamp}.log
        elif [[ $CHANNELS == "last" ]]; then
            # channels last
            echo "Channels last testing...."
            python benchmarks/dynamo/runner.py --dashboard-archive-path=${log_path}/archive --devices=cpu --dtypes=float32 --inference --suite=${SUITE} --compilers=inductor --channels-last --batch_size=${BS} --output-dir=${log_path}/multi_threads_cl_logs_${timestamp} 2>&1 | tee ${log_path}/multi_threads_${SUITE}_model_bench_bs_${BS}_log_${timestamp}.log
        else
            echo "Please check channels foramt with first / last."
        fi
    done
}

multi_threads_tune
