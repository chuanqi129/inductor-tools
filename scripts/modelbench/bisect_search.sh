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
    # git reset --hard HEAD && git checkout main && git checkout ${END_COMMIT} && git submodule sync && git submodule update --init --recursive
    # python setup.py clean && python setup.py develop
    git checkout ${END_COMMIT}
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $PRECISION $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    expected_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Expected performance: $expected_perf s"
fi
chmod +x bisect_run_test.sh
git reset --hard HEAD && git checkout ${TORCH_BRANCH} && git pull && git checkout ${START_COMMIT}
git bisect start ${START_COMMIT} ${END_COMMIT}
git bisect run ./bisect_run_test.sh $SUITE $MODEL $MODE $PRECISION $SHAPE $WRAPPER $SCENARIO $KIND $THREADS $CHANNELS $FREEZE 0 $expected_perf $BACKEND $PERF_RATIO $EXTRA 2>&1 | tee ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log
git bisect reset
cat ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log | grep "is the first bad commit" | awk '{ print $1 }' > ${LOG_DIR}/guitly_commit.log
echo "Start commit: ${START_COMMIT}" >> ${LOG_DIR}/guitly_commit.log
echo "End commit: ${END_COMMIT}" >> ${LOG_DIR}/guitly_commit.log
