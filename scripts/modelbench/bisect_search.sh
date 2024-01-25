#!/bin/bash
set +x
TORCH_BRANCH=${1:-main}
START_COMMIT=${2:-main}
END_COMMIT=${3:-main}
SUITE=${4:-torchbench}
MODEL=${5:-resnet50}
MODE=${6:-inference}
SCENARIO=${7:-accuracy}
PRECISION=${8:-float32}
SHAPE=${9:-static}
WRAPPER=${10:-default}
KIND=${11:-crash}
THREADS=${12:-multiple}
CHANNELS=${13:-first}
FREEZE=${14:-on}
BS=${15:0}
LOG_DIR=${16:-inductor_log}
HF_TOKEN=${17:-hf_xxx}
BACKEND=${18:-inductor}
PERF_RATIO=${19:-1.1}
EXTRA=${20}

export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

echo "===============Start searching guilty commit for ${SUITE} ${MODEL} ${MODE} ${PRECISION} ${SHAPE} ${WRAPPER} ${SCENARIO} ${KIND}================="
cd /workspace/pytorch
expected_perf=0
# For perfroamcen drop issue, get the expected performance based on end/good commit
if [ "$SCENARIO" == "performance" ] && [ "$KIND" == "drop" ]; then
    # Initial image build with END_COMMIT, no need rebuild
    git checkout ${END_COMMIT}
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    expected_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Expected performance: $expected_perf s" > ${LOG_DIR}/perf_drop.log

    # Check START_COMMIT performance for early stop
    rm -rf /tmp/*
    git reset --hard ${START_COMMIT} && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop && cd .. && \
    rm -rf vision && git clone -b main https://github.com/pytorch/vision.git && cd vision && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt` && pip uninstall torchvision -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf data && git clone -b main https://github.com/pytorch/data.git && cd data && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`  && pip uninstall torchdata -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf text && git clone -b main https://github.com/pytorch/text.git && cd text && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt` && pip uninstall torchtext -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf audio && git clone -b main https://github.com/pytorch/audio.git && cd audio && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt` && pip uninstall torchaudio -y && python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    cd benchmark && git checkout main && git checkout `cat /workspace/pytorch/.github/ci_commit_pins/torchbench.txt` && cd /workspace/pytorch
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    current_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Current performance: $current_perf s" >> ${LOG_DIR}/perf_drop.log

    ratio=$(echo "$current_perf $expected_perf" | awk '{ printf "%.2f\n", $1/$2 }')
    echo "=====ratio: $ratio======="
    if (( $(echo "$ratio > $PERF_RATIO" | bc -l) )); then
        echo "Real performance drop issue, will search guilty commit." >> ${LOG_DIR}/perf_drop.log
    else
        echo "Fake performance drop issue, please double check it." >> ${LOG_DIR}/perf_drop.log
        exit 0
    fi
fi
chmod +x bisect_run_test.sh
git reset --hard && git pull && git checkout ${START_COMMIT}
git bisect start ${START_COMMIT} ${END_COMMIT}
git bisect run ./bisect_run_test.sh $SUITE $MODEL $MODE $PRECISION $SHAPE $WRAPPER $SCENARIO $KIND $THREADS $CHANNELS $FREEZE 0 $expected_perf $BACKEND $PERF_RATIO $EXTRA 2>&1 | tee ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log
git bisect reset
cat ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log | grep "is the first bad commit" | awk '{ print $1 }' > ${LOG_DIR}/guitly_commit.log
echo "Start commit: ${START_COMMIT}" >> ${LOG_DIR}/guitly_commit.log
echo "End commit: ${END_COMMIT}" >> ${LOG_DIR}/guitly_commit.log
