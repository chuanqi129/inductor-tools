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
LOG_DIR=${5:-inductor_log}
# default / cpp
WRAPPER=${6:-default}
HF_TOKEN=${7:-hf_xx}
BACKEND=${8:-inductor}
# Test Mode
TEST_MODE=${9:-inference}
SUITE=${10:-all}
# Easy to enbale new test
EXTRA=${11}

mkdir -p $LOG_DIR

export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
# https://github.com/pytorch/pytorch/issues/107200
pip uninstall transformers -y && pip install transformers==4.30.2
# fix issue: AttributeError: module 'importlib.resources' has no attribute 'files'
pip uninstall networkx -y && pip install networkx

# skip sam & nanogpt_generate for stable results
sed -i '/skip_str = " ".join(skip_tests)/a\    skip_str += " -x sam -x nanogpt_generate"' benchmarks/dynamo/runner.py

if [[ ${TEST_MODE} == "training_full" ]]; then
    # skip hf_GPT2_large, cuz it will OOM after using jemalloc
    sed -i "/SKIP_TRAIN = {/a\    \"hf_GPT2_large\"," benchmarks/dynamo/torchbench.py
fi

Flag_extra="$EXTRA "
if [[ $BACKEND == "aot_inductor" ]]; then
    echo "Testing with aot_inductor."
    # Workaround for test with runner.py
    sed -i '/"inference": {/a \ \ \ \ \ \ \ \ "aot_inductor": "--inference -n50 --export-aot-inductor ",' benchmarks/dynamo/runner.py
elif [[ $BACKEND == "inductor" ]]; then
    echo "Setting freezing for inductor backend by default."
    export TORCHINDUCTOR_FREEZING=1
    Flag_extra+="--freezing "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

# Wrapper_extra=""
if [[ $WRAPPER == "cpp" ]]; then
    echo "Testing with cpp wrapper."
    export TORCHINDUCTOR_CPP_WRAPPER=1
fi

if [[ $SUITE == "all" ]]; then
    SUITE=""
else
    SUITE="--suites=${SUITE}"
fi

# multi-threads
multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [[ $CHANNELS == "first" ]]; then
        # channels first
        echo "Channels first testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --${TEST_MODE} --compilers=${BACKEND} $SUITE --extra-args="--timeout 9000 ${Shape_extra} ${Flag_extra}" --output-dir=${LOG_DIR}/multi_threads_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --${TEST_MODE} --compilers=${BACKEND} $SUITE --extra-args="--timeout 9000 ${Shape_extra} ${Flag_extra}" --channels-last --output-dir=${LOG_DIR}/multi_threads_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
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
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --${TEST_MODE} --compilers=${BACKEND} $SUITE --batch_size=1 --threads 1 --extra-args="--timeout 9000 ${Shape_extra} ${Wrapper_extra} ${Flag_extra}" --output-dir=${LOG_DIR}/single_thread_cf_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
    elif [[ $CHANNELS == "last" ]]; then
        # channels last
        echo "Channels last testing...."
        python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --dashboard-archive-path=${LOG_DIR}/archive --devices=cpu --dtypes=${DT} --${TEST_MODE} --compilers=${BACKEND} $SUITE --channels-last --batch_size=1 --threads 1 --extra-args="--timeout 9000 ${Shape_extra} ${Wrapper_extra} ${Flag_extra}" --output-dir=${LOG_DIR}/single_thread_cl_logs_${timestamp} 2>&1 | tee ${LOG_DIR}/single_thread_model_bench_log_${timestamp}.log
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
