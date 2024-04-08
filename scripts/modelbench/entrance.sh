#!/bin/bash
set -xe

# set params firstly
# default
TAG="ww18.4"
PRECISION="float32"
TEST_MODE="inference"
SHAPE="static"
TORCH_REPO="https://github.com/pytorch/pytorch.git"
TORCH_COMMIT="nightly"
DYNAMO_BENCH="fea73cb"
AUDIO="0a652f5"
TEXT="c4ad5dd"
VISION="f2009ab"
DATA="5cb3e6d"
TORCH_BENCH="a0848e19"
THREADS="all"
CHANNELS="first"
WRAPPER="default"
HF_TOKEN=""
BACKEND="inductor"
SUITE="all"
MODEL="resnet50"
TORCH_START_COMMIT="nightly"
TORCH_END_COMMIT="${TORCH_START_COMMIT}"
SCENARIO="accuracy"
KIND="crash" # issue kind crash/drop
PERF_RATIO="-1.1"
EXTRA=""
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

# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

# rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt file
if [ -f finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt ]; then
    rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi

# launch benchmark
bash launch.sh \
    TAG=${TAG} \
    PRECISION=${PRECISION} \
    TEST_MODE=${TEST_MODE} \
    TEST_SHAPE=${SHAPE} \
    TORCH_REPO=${TORCH_REPO} \
    TORCH_COMMIT=${TORCH_COMMIT} \
    DYNAMO_BENCH=${DYNAMO_BENCH} \
    AUDIO=${AUDIO} \
    TEXT=${TEXT} \
    VISION=${VISION} \
    DATA=${DATA} \
    TORCH_BENCH=${TORCH_BENCH} \
    THREADS=${THREADS} \
    CHANNELS=${CHANNELS} \
    WRAPPER=${WRAPPER} \
    HF_TOKEN=${HF_TOKEN} \
    BACKEND=${BACKEND} \
    SUITE=${SUITE} \
    MODEL=${MODEL} \
    TORCH_START_COMMIT=${TORCH_START_COMMIT} \
    TORCH_END_COMMIT=${TORCH_END_COMMIT} \
    SCENARIO=${SCENARIO} \
    KIND=${KIND} \
    PERF_RATIO=${PERF_RATIO} \
    EXTRA=${EXTRA}

# create finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    echo "Finished!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
else
    echo "benchmark failed!"
    echo "Failed!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi
