export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export USE_LLVM=False

# multiple / single / all
THREAD=${1:-all}
# first / last
CHANNELS=${2:-first}
# float32 / bfloat16
DT=${3:-float32}
# static / dynamic
SHAPE=${4:-static}
LOG_DIR=${5:-ipex_log}
DYNAMO_BENCH=${6:-1238ae3}
mkdir -p $LOG_DIR

# collect sw info
curdir=$(pwd)
touch ${curdir}/${LOG_DIR}/version.txt
cd /workspace/benchmark
echo torchbench : $(git rev-parse --short HEAD) >>${curdir}/${LOG_DIR}/version.txt
cd /workspace/pytorch
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata; \
	print("torch : ", torch.__version__); \
	print("torchvision : ", torchvision.__version__); \
	print("torchtext : ", torchtext.__version__); \
	print("torchaudio : ", torchaudio.__version__); \
	print("torchdata : ", torchdata.__version__)''' >>${curdir}/${LOG_DIR}/version.txt
echo dynamo_benchmarks : $DYNAMO_BENCH >>${curdir}/${LOG_DIR}/version.txt

sed -i "s%torch._dynamo.config.base_dir%'/workspace/pytorch'%" ./benchmarks/dynamo/common.py
sed -i '/\"inductor\": \"--inference -n50 --inductor \"/a\"ipex\": \"--backend=ipex --inference \",' ./benchmarks/dynamo/runner.py

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes "
fi

# multi-threads
multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [[ $CHANNELS == "first" ]]; then
        # channels first
        echo "Channels first testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --inference  --compilers=ipex --extra-args="--timeout 9000 ${Shape_extra}" --output-dir=${LOG_DIR}/multi_threads_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --inference  --compilers=ipex --extra-args="--timeout 9000 ${Shape_extra}" --channels-last --output-dir=${LOG_DIR}/multi_threads_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
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
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --inference --compilers=ipex --batch_size=1 --threads 1 --extra-args="--timeout 9000 ${Shape_extra}" --output-dir=${LOG_DIR}/single_thread_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --inference --compilers=ipex --channels-last --batch_size=1 --threads 1 --extra-args="--timeout 9000 ${Shape_extra}" --output-dir=${LOG_DIR}/single_thread_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
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
