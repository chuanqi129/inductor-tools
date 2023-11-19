#!/bin/bash

SUITE=${1:-huggingface}	# torchbench / huggingface / timm_models
MODEL=${2:-GoogleFnet}
MODE=${3:-inference} # inference / training
DT=${4:-float32} # float32 / amp
SHAPE=${5:-static} # static / dynamic
SHAPE=${6:-static} # static / dynamic
KIND=${7:-drop} # crash / drop
THREADS=${8:-multiple} # multiple / single
CHANNELS=${9:-first} # first / last
FREEZE=${10:-on} # on / off
BS=${11:-0} # default / specific
EXP_PERF=${12:-0} 
BACKEND=${13:-inductor}
EXTRA=${14}

prepare_test() {
    git reset --hard HEAD && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop
}

run_perf_drop_test() {
    detected_value=$(bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "=====result: $result======="
    ratio=$(echo "$result $EXP_PERF" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if [[ $ratio > 1.1 ]]; then
        exit 1
    else
        exit 0
    fi
}

run_acc_drop_test() {
    acc_res=$(bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $4}')
    echo "=====acc: $acc_res======="
    if [ $acc_res != "pass" ]; then
        exit 1
    else
        exit 0
    fi
}

run_crash_test() {
    bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE
    if [ $? -eq 0 ]; then
        exit 1
    else
        exit 0
    fi
}

prepare_test
if [ $KIND == "drop" ]; then
    if [ $SCENARIO == "performance" ]; then
        run_perf_drop_test
    else
        run_acc_drop_test
    fi
else
    run_crash_test
fi
