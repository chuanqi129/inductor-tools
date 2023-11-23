#!/bin/bash
EXPECTED_PERFORMANCE=${1}
THREAD=${2:-multiple} # multiple / single / all
MODE=${3:-inference} # inference / training
SCENARIO=${4:-accuracy} # accuracy / performance
SUITE=${5:-huggingface} # torchbench / huggingface / timm_models
MODEL=${6:-GoogleFnet}
DT=${7:-float32} # float32 / amp
CHANNELS=${8:-first} # first / last
SHAPE=${9:-static} # static / dynamic
WRAPPER=${10:-default} # default / cpp
BS=${11:-0}
FREEZE=${12:-on}

run_test() {
    git reset --hard HEAD && git checkout 0200b11 benchmarks
    python setup.py clean && python setup.py develop
    detected_value=$(bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "=====result: $result======="
    ratio=$(echo "$result $EXPECTED_PERFORMANCE" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    git reset --hard HEAD
    if [[ $ratio > 1.1 ]]; then
        exit 1
    else
        exit 0
    fi
}

run_test
