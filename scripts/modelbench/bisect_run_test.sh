#!/bin/bash

SUITE=${1:-huggingface}	# torchbench / huggingface / timm_models
MODEL=${2:-GoogleFnet}
MODE=${3:-inference} # inference / training
DT=${4:-float32} # float32 / amp
SHAPE=${5:-static} # static / dynamic
WRAPPER=${6:-default} # default / cpp
SCENARIO=${7:-accuracy} # accuracy / performance
KIND=${8:-drop} # crash / drop / fixed / improve
THREADS=${9:-multiple} # multiple / single
CHANNELS=${10:-first} # first / last
BS=${11:-0} # default / specific
EXP_PERF=${12:-0}
BACKEND=${13:-inductor}
PERF_RATIO=${14:-1.1} # default 10% perforamnce drop regard as issue
EXTRA=${15}

prepare_test() {
    rm -rf /tmp/*
    git reset --hard HEAD && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop && cd .. && \
    cd vision && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt` && pip uninstall torchvision -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    # cd data && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`  && pip uninstall torchdata -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    cd text && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt` && pip uninstall torchtext -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    # cd audio && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt` && pip uninstall torchaudio -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    export TRANSFORMERS_COMMIT=`cat /workspace/pytorch/.ci/docker/ci_commit_pins/huggingface.txt` && pip install --force-reinstall git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}
    pip install numpy==1.26.4
    cd benchmark && git checkout main && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/torchbench.txt` && cd /workspace/pytorch
}

run_perf_drop_test() {
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND | tail -n 1 | awk -F, '{print $5}')
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "=====result: $result======="
    ratio=$(echo "$result $EXP_PERF" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if (( $(echo "$ratio > $PERF_RATIO" | bc -l) )); then
	    echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 1
    else
	    echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 0
    fi
}

run_perf_improve_test() {
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND | tail -n 1 | awk -F, '{print $5}')
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "=====result: $result======="
    ratio=$(echo "$EXP_PERF $result" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if (( $(echo "$ratio > $PERF_RATIO" | bc -l) )); then
	    echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 1
    else
	    echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 0
    fi
}

run_acc_drop_test() {
    acc_res=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND | tail -n 1 | awk -F, '{print $4}')
    echo "=====acc: $acc_res======="
    if [ "X$acc_res" != "Xpass" ]; then
	    echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 1
    else
	    echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 0
    fi
}

run_acc_improve_test() {
    acc_res=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND | tail -n 1 | awk -F, '{print $4}')
    echo "=====acc: $acc_res======="
    if [ "X$acc_res" != "Xpass" ]; then
	    echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 0
    else
	    echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 1
    fi
}

run_crash_test() {
    bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND 2>&1 | tee ./crash.log
    if [ $? -eq 0 ]; then
        acc_status=`tail -n 1 ./crash.log | grep pass | wc -l`
        perf_status=`tail -n 1 ./crash.log | grep $MODEL | awk -F, '{print $3}'`
        echo $acc_status
        echo $perf_status
        if [ "$SCENARIO" == "accuracy" ]; then
            if [ $acc_status -eq 0 ]; then
                echo "`git rev-parse HEAD` is a BAD COMMIT!"
                exit 1
            else
                echo "`git rev-parse HEAD` is a GOOD COMMIT!"
                exit 0
            fi
        elif [ "$SCENARIO" == "performance" ]; then
            if [[ ! -z $perf_status ]] && [ $perf_status -gt 0 ]; then
                echo "`git rev-parse HEAD` is a GOOD COMMIT!"
                exit 0
            else
                echo "`git rev-parse HEAD` is a BAD COMMIT!"
                exit 1
            fi
        fi
    else
        echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 1
    fi
}

run_fixed_test() {
    bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $BACKEND 2>&1 | tee ./fixed.log
    if [ $? -eq 0 ]; then
        acc_status=`tail -n 1 ./fixed.log | grep pass | wc -l`
        perf_status=`tail -n 1 ./fixed.log | grep $MODEL | awk -F, '{print $3}'`
        echo $acc_status
        echo $perf_status
        if [ "$SCENARIO" == "accuracy" ]; then
            if [ $acc_status -eq 0 ]; then
                echo "`git rev-parse HEAD` is a BAD COMMIT!"
                exit 0
            else
                echo "`git rev-parse HEAD` is a GOOD COMMIT!"
                exit 1
            fi
        elif [ "$SCENARIO" == "performance" ]; then
            if [[ ! -z $perf_status ]] && [ $perf_status -gt 0 ]; then
                echo "`git rev-parse HEAD` is a GOOD COMMIT!"
                exit 1
            else
                echo "`git rev-parse HEAD` is a BAD COMMIT!"
                exit 0
            fi
        fi
    else
        echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 1
    fi
}

prepare_test > inductor_log/bisect_pt_build.log 2>&1

if [ $? -eq 0 ]; then
    echo "PT build success!"
else
    echo "PT build failed!"
fi

cd /workspace/pytorch
if [ $KIND == "drop" ]; then
    if [ $SCENARIO == "performance" ]; then
        run_perf_drop_test
    else
        run_acc_drop_test
    fi
elif [ $KIND == "improve" ]; then
    if [ $SCENARIO == "performance" ]; then
        run_perf_improve_test
    else
        run_acc_improve_test
    fi
elif [ $KIND == "crash" ]; then
    run_crash_test
elif [ $KIND == "fixed" ]; then
    run_fixed_test
fi
