#!/bin/bash
set -xe

START_COMMIT="main"
END_COMMIT="main"
SUITE="torchbench"
MODEL="resnet50"
MODE="inference"
SCENARIO="accuracy"
PRECISION="float32"
SHAPE="static"
WRAPPER="default"
KIND="crash"
THREADS="multiple"
CHANNELS="first"
FREEZE="on"
BS="0"
LOG_DIR="inductor_log"
HF_TOKEN=""
BACKEND="inductor"
PERF_RATIO="-1.1"
EXTRA=
# get value from param
if [[ "$@" != "" ]];then
    echo "" > tmp.env
    for var in "$@"
    do
        if [[ "${var}" == "EXTRA="* ]];then
            EXTRA="${@/*EXTRA=}"
            break
        else
            echo "$var" >> tmp.env
        fi
        shift
    done
    source tmp.env && rm -rf tmp.env
fi

export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

echo "===============Start searching guilty commit for ${SUITE} ${MODEL} ${MODE} ${PRECISION} ${SHAPE} ${WRAPPER} ${SCENARIO} ${KIND}================="
cd /workspace/pytorch
expected_perf=0
# For perfroamcen drop issue, get the expected performance based on end/good commit
if [ "$SCENARIO" == "performance" ] && ([ "$KIND" == "drop" ] || [ "$KIND" == "improve" ]); then
    # Initial image build with END_COMMIT, no need rebuild
    git checkout ${END_COMMIT}
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER $BS $FREEZE $BACKEND | tail -n 1 | awk -F, '{print $5}')
    expected_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Expected performance: $expected_perf s" > ${LOG_DIR}/perf_drop.log

    # Check START_COMMIT performance for early stop
    rm -rf /tmp/*
    git reset --hard HEAD && git checkout ${START_COMMIT} && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop && cd .. && \
    cd vision && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt` && pip uninstall torchvision -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    cd data && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`  && pip uninstall torchdata -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    cd text && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt` && pip uninstall torchtext -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    cd audio && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt` && pip uninstall torchaudio -y && python setup.py bdist_wheel && pip install dist/*.whl && cd /workspace/pytorch && \
    # export TRANSFORMERS_COMMIT=`cat /workspace/pytorch/.ci/docker/ci_commit_pins/huggingface.txt` && pip install --force-reinstall git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT} && cd /workspace/pytorch
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER $BS $FREEZE $BACKEND | tail -n 1 | awk -F, '{print $5}')
    current_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Current performance: $current_perf s" >> ${LOG_DIR}/perf_drop.log

    if [ "$KIND" == "drop" ]; then
        ratio=$(echo "$current_perf $expected_perf" | awk '{ printf "%.2f\n", $1/$2 }')    
    elif [ "$KIND" == "improve" ]; then
        ratio=$(echo "$expected_perf $current_perf" | awk '{ printf "%.2f\n", $1/$2 }')    
    fi
    echo "=====ratio: $ratio======="
    if (( $(echo "$ratio > $PERF_RATIO" | bc -l) )); then
        echo "Real performance ${KIND}, will search guilty commit." >> ${LOG_DIR}/perf_drop.log
    else
        echo "Fake performance ${KIND}, please double check it." >> ${LOG_DIR}/perf_drop.log
        exit 0
    fi
fi
chmod +x bisect_run_test.sh
git reset --hard HEAD &&  git fetch origin -a && git checkout ${START_COMMIT}
git bisect start ${START_COMMIT} ${END_COMMIT}
git bisect run ./bisect_run_test.sh $SUITE $MODEL $MODE $PRECISION $SHAPE $WRAPPER $SCENARIO $KIND $THREADS $CHANNELS $FREEZE 0 $expected_perf $BACKEND $PERF_RATIO $EXTRA 2>&1 | tee ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${THREADS}-${SCENARIO}-${KIND}_guilty_commit.log
git bisect reset
cat ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${THREADS}-${SCENARIO}-${KIND}_guilty_commit.log | grep "is the first bad commit" | awk '{ print $1 }' > ${LOG_DIR}/guilty_commit.log
echo "Start commit: ${START_COMMIT}" >> ${LOG_DIR}/guilty_commit.log
echo "End commit: ${END_COMMIT}" >> ${LOG_DIR}/guilty_commit.log
