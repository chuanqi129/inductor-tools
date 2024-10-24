export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

THREAD=${1:-multiple} # multiple / single / all
MODE=${2:-inference} # inference / training
SCENARIO=${3:-accuracy} # accuracy / performance
SUITE=${4:-huggingface} # torchbench / huggingface / timm_models
MODEL=${5:-GoogleFnet}
DT=${6:-float32} # float32 / amp
CHANNELS=${7:-first} # first / last
SHAPE=${8:-static} # static / dynamic
WRAPPER=${9:-default} # default / cpp
BS=${10:-0}
FREEZE=${11:-on}

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
    Wrapper_extra="--cpp_wrapper "
    export TORCHINDUCTOR_CPP_WRAPPER=1
fi

if [[ $WRAPPER == "ptq" ]]; then
    echo "Testing with PTQ."
fi

if [[ $WRAPPER == "qat" ]]; then
    echo "Testing with QAT."
    Wrapper_extra="--is_qat "
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
if [[ ${FREEZE} == "on" ]]; then
    export TORCHINDUCTOR_FREEZING=1
    echo "Testing with freezing on."
    Flag_extra="--freezing " 
fi 
cd ../benchmark
rm -rf .userbenchmark/
TORCHINDUCTOR_CPP_WRAPPER=1 python run_benchmark.py cpu -m ${MODEL} --torchdynamo inductor --quantize --cpp_wrapper --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs
cat .userbenchmark/cpu/*.json | grep "throughput" | awk -F':' '{print $2}'
