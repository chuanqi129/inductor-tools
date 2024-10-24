#!/bin/bash

SUITE=${1:-huggingface}	# torchbench / huggingface / timm_models
MODEL=${2:-GoogleFnet}
MODE=${3:-inference} # inference / training
DT=${4:-float32} # float32 / amp
SHAPE=${5:-static} # static / dynamic
WRAPPER=${6:-default} # default / cpp
SCENARIO=${7:-accuracy} # accuracy / performance
KIND=${8:-drop} # crash / drop
THREADS=${9:-multiple} # multiple / single
CHANNELS=${10:-first} # first / last
FREEZE=${11:-on} # on / off
BS=${12:-0} # default / specific
EXP_PERF=${13:-0}
BACKEND=${14:-inductor}
PERF_RATIO=${15:-1.1} # default 10% perforamnce drop regard as issue
EXTRA=${16}

prepare_test() {
    rm -rf /tmp/*
    git reset --hard HEAD && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop && cd .. && \
    rm -rf vision && git clone -b main https://github.com/pytorch/vision.git && cd vision && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt` && pip uninstall torchvision -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf data && git clone -b main https://github.com/pytorch/data.git && cd data && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`  && pip uninstall torchdata -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf text && git clone -b main https://github.com/pytorch/text.git && cd text && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt` && pip uninstall torchtext -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf audio && git clone -b main https://github.com/pytorch/audio.git && cd audio && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt` && pip uninstall torchaudio -y && python setup.py bdist_wheel && pip install dist/*.whl && cd /workspace/pytorch 
    # cd benchmark && git checkout main && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/torchbench.txt` && cd /workspace/pytorch
}

run_perf_drop_test() {
    # detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    detected_value=$(bash ./quant_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER | tail -n 1)
    result=$(echo $detected_value | awk '{ printf "%.5f", $1/1 }')
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
    acc_res=$(bash hf_quant_test.sh key torch_compile_quant_static | grep "eval_accuracy" | awk -F'=' '{print $2}')
    echo "=====acc: $acc_res======="
    ratio=$(echo "$EXP_PERF $acc_res" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if (( $(echo "$ratio > $PERF_RATIO" | bc -l) )); then
	    echo "`git rev-parse HEAD` is a BAD COMMIT!"
        exit 1
    else
	    echo "`git rev-parse HEAD` is a GOOD COMMIT!"
        exit 0
    fi
}

run_crash_test() {
    # bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE 2>&1 | tee ./crash.log
    # python inductor_quant_acc.py 2>&1 | tee ./crash.log
    # bash hf_quant_test.sh key torch_compile_quant_static 2>&1 | tee ./crash.log
    cd ../benchmark
    TORCHINDUCTOR_CPP_WRAPPER=1 TORCHINDUCTOR_FREEZING=1 python run_benchmark.py cpu -m $MODEL --torchdynamo inductor --quantize --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs 2>&1 | tee ./crash.log
    if [ $? -eq 0 ]; then
        # acc_status=`tail -n 1 ./crash.log | grep int8 | wc -l`
	# acc_status=`cat ./crash.log | grep "eval_accuracy" | wc -l`
 	acc_status=`cat ./crash.log | grep "Done" | wc -l`
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
else
    run_crash_test
fi
