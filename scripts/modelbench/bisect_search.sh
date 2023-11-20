#!/bin/bash

START_COMMIT=${1:-main}
END_COMMIT=${2:-main}
SUITE=${3:-torchbench}
MODEL=${4:-resnet50}
MODE=${5:-inference}
SCENARIO=${6:-accuracy}
PRECISION=${7:-float32}
SHAPE=${8:-static}
WRAPPER=${9:-default}
KIND=${10:-crash}
THREADS=${11:-multiple}
CHANNELS=${12:-first}
FREEZE=${13:-on}
BS=${14:0}
LOG_DIR=${15:-inductor_log}
HF_TOKEN=${16:-hf_xxx}
BACKEND=${17:-inductor}
EXTRA=${18}

export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

echo "===============Start searching guilty commit for ${SUITE} ${MODEL} ${MODE} ${PRECISION} ${SHAPE} ${WRAPPER} ${SCENARIO} ${KIND}================="
cd /workspace/pytorch
expected_perf=0
# For perfroamcen drop issue, get the expected performance based on end/good commit
if [ $SCENARIO == "performacne" ] && [ $KIND == "drop" ]; then
    git reset --hard HEAD && git checkout main && git checkout ${START_COMMIT} && git submodule sync && git submodule update --init --recursive
    python setup.py clean && python setup.py develop
    detected_value=$(bash ./inductor_single_run.sh $THREADS $MODE $SCENARIO $SUITE $MODEL $DT $CHANNELS $SHAPE $WRAPPER $BS $FREEZE | tail -n 1 | awk -F, '{print $5}')
    expected_perf=$(echo $detected_value | awk '{ printf "%.5f", $1/1000 }')
    echo "Expected performance: $expected_perf s"
fi
chmod +x bisect_run_test.sh
git reset --hard HEAD && git checkout main && git pull && git checkout ${START_COMMIT}
git bisect start ${START_COMMIT} ${END_COMMIT}
git bisect run ./bisect_run_test.sh $SUITE $MODEL $MODE $PRECISION $SHAPE $WRAPPER $SCENARIO $KIND $THREADS $CHANNELS $FREEZE 0 $expected_perf $BACKEND $EXTRA 2>&1 | tee ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log
git bisect reset
guilty_commit=`cat ${LOG_DIR}/${SUITE}-${MODEL}-${MODE}-${PRECISION}-${SHAPE}-${WRAPPER}-${SCENARIO}-${THREADS}-${KIND}_guilty_commit.log | grep "bisect found first" | awk -F' ' '{ print $9 }'`
echo ${guitly_commit}
