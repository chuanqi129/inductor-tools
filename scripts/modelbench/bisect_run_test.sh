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
    python setup.py clean && python setup.py develop && cd ..
    rm -rf vision && git clone -b main https://github.com/pytorch/vision.git && cd vision && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt` && pip uninstall torchvision -y && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
    rm -rf data && git clone -b main https://github.com/pytorch/data.git && cd data && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`  && pip uninstall torchdata -y && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
    rm -rf text && git clone -b main https://github.com/pytorch/text.git && cd text && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt` && pip uninstall torchtext -y && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
    rm -rf audio && git clone -b main https://github.com/pytorch/audio.git && cd audio && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt` && pip uninstall torchaudio -y && python setup.py bdist_wheel && pip install dist/*.whl && cd ..
    cd benchmark && hit checkout main && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/torchbench.txt`
}

run_perf_drop_test() {
    detected_value=$(bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "=====result: $result======="
    ratio=$(echo "$result $EXP_PERF" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if [[ $ratio > 1.1 ]]; then
	echo "BAD COMMIT!"
        exit 1
    else
	echo "GOOD COMMIT!"
        exit 0
    fi
}

run_acc_drop_test() {
    acc_res=$(bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $4}')
    echo "=====acc: $acc_res======="
    if [ $acc_res != "pass" ]; then
	echo "BAD COMMIT!"
        exit 1
    else
	echo "GOOD COMMIT!"
        exit 0
    fi
}

run_crash_test() {
    bash ./inductor_single_run.sh $THREAD $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE
    if [ $? -eq 0 ]; then
	echo "GOOD COMMIT!"
        exit 0
    else
	echo "BAD COMMIT!"
        exit 1
    fi
}

prepare_test > inductor_log/bisect_pt_build.log 2>&1

if [ $? -eq 0 ]; then
    echo "PT build success!"
else
    echo "PT build failed!"
fi

if [ $KIND == "drop" ]; then
    if [ $SCENARIO == "performance" ]; then
        run_perf_drop_test
    else
        run_acc_drop_test
    fi
else
    run_crash_test
fi
