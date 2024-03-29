#! /bin/bash
# This script work for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
DT=${2:-float32}            # float32 / float16 / amp_fp16 / amp_bf16
MODE=${3:-inference}        # inference / training
SCENARIO=${4:-accuracy}     # accuracy / performance
DEVICE=${5:-xpu}            # xpu / cuda
CARD=${6:-0}                # 0 / 1 / 2 / 3 ...
GRAPH=${7:-on}		    # on / off
SHAPE=${8:-static}          # static / dynamic

WORKSPACE=`pwd`
LOG_DIR=$WORKSPACE/inductor_log/${SUITE}/${DT}
mkdir -p ${LOG_DIR}
LOG_NAME=inductor_${SUITE}_${DT}_${MODE}_${DEVICE}_${SCENARIO}

if [[ $DT == "amp_bf16" ]]; then
    DT="amp"
elif [[ $DT == "amp_fp16" ]]; then
    DT="amp"
    export INDUCTOR_AMP_DT=float16
fi

Cur_Ver=`pip list | grep "^torch " | awk '{print $2}' | cut -d"+" -f 1`
if [ $(printf "${Cur_Ver}\n2.0.2"|sort|head -1) = "${Cur_Ver}" ]; then
    Mode_extra="";
else
    # For PT 2.1
    Mode_extra="--inference --freezing ";
fi
if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

Cuda_extra=""
if [[ $GRAPH == "off" ]]; then
    echo "Testing without cudagraph."
    Cuda_extra="--disable-cudagraphs "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

ulimit -n 1048576
ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -d${DEVICE} -n50 --no-skip --dashboard ${Mode_extra} ${Shape_extra} ${Cuda_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
