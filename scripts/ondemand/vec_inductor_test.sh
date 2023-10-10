export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

THREAD=${1:-multiple} # multiple / single / all
MODE=${2:-inference} # inference / training
SCENARIO=${3:-accuracy} # accuracy / performance
SUITE=${4:-huggingface} # torchbench / huggingface / timm_models
DT=${5:-float32} # float32 / amp
CHANNELS=${6:-first} # first / last
SHAPE=${7:-static} # static / dynamic
WRAPPER=${8:-default} # default / cpp
BS=${9:-0}

Mode_extra="--inference "
if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

Wrapper_extra=""
if [[ $WRAPPER == "cpp" ]]; then
    echo "Testing with cpp wrapper."
    Wrapper_extra="--cpp-wrapper "
fi

Channels_extra=""
if [[ ${CHANNELS} == "last" ]]; then
    Channels_extra="--channels-last "
fi

BS_extra=""
if [[ ${BS} -gt 0 ]]; then
    BS_extra="--batch_size=${BS} "
fi

Flag_extra=""
export TORCHINDUCTOR_FREEZING=1
echo "Testing with freezing on."
Flag_extra="--freezing "    



multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    end_core=$(expr $CORES - 1)    
    numactl -C 0-${end_core} --membind=0 python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -dcpu -n50 --no-skip --dashboard ${Channels_extra} ${BS_extra} ${Shape_extra} ${Mode_extra} ${Wrapper_extra} ${Flag_extra} --timeout 9000 --backend=inductor  --output=/tmp/inductor_single_test_mt.csv
    cat /tmp/inductor_single_test_mt.csv && rm /tmp/inductor_single_test_mt.csv
}

single_thread_test() {
    export OMP_NUM_THREADS=1
    numactl -C 0-0 --membind=0 python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -dcpu -n50 --no-skip --dashboard --batch-size 1 --threads 1 ${Channels_extra} ${Shape_extra} ${Mode_extra} ${Wrapper_extra} ${Flag_extra} --timeout 9000 --backend=inductor  --output=/tmp/inductor_single_test_st.csv
    cat /tmp/inductor_single_test_st.csv && rm /tmp/inductor_single_test_st.csv
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
