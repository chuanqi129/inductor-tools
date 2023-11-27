#!/bin/bash
set -xe

# set params firstly
# default
TAG="ww18.4"
PRECISION="float32"
TEST_MODE="inference"
TEST_SHAPE="static"
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

echo "TAG" : $TAG
echo "PRECISION" : $PRECISION
echo "TEST_MODE" : $TEST_MODE
echo "TEST_SHAPE" : $TEST_SHAPE
echo "TORCH_REPO" : $TORCH_REPO
echo "TORCH_BRANCH" : $TORCH_BRANCH
echo "TORCH_COMMIT" : $TORCH_COMMIT
echo "DYNAMO_BENCH" : $DYNAMO_BENCH
echo "AUDIO" : $AUDIO
echo "TEXT" : $TEXT
echo "VISION" : $VISION
echo "DATA" : $DATA
echo "TORCH_BENCH" : $TORCH_BENCH
echo "THREADS" : $THREADS
echo "CHANNELS" : $CHANNELS
echo "WRAPPER" : $WRAPPER
echo "BACKEND" : $BACKEND
echo "SUITE" : $SUITE
echo "MODEL" : $MODEL
echo "TORCH_START_COMMIT" : $TORCH_START_COMMIT
echo "TORCH_END_COMMIT" : $TORCH_END_COMMIT
echo "SCENARIO" : $SCENARIO
echo "KIND" : $KIND

# clean up
docker stop $(docker ps -aq) || true
docker system prune -af

LOG_DIR="inductor_log"
if [ -d ${LOG_DIR} ]; then
    sudo rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}

DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=$TORCH_REPO --build-arg PT_COMMIT=$TORCH_COMMIT --build-arg BENCH_COMMIT=$DYNAMO_BENCH --build-arg TORCH_AUDIO_COMMIT=$AUDIO --build-arg TORCH_TEXT_COMMIT=$TEXT --build-arg TORCH_VISION_COMMIT=$VISION --build-arg TORCH_DATA_COMMIT=$DATA --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH --build-arg https_proxy=${https_proxy} --build-arg HF_HUB_TOKEN=$HF_TOKEN -t pt_inductor:$TAG -f Dockerfile --target image . > ${LOG_DIR}/image_build.log 2>&1
docker_build_result=${PIPESTATUS[0]}
# Early exit for docker image build issue
if [ "$docker_build_result" != "0" ];then
    echo "Docker image build failed, early exit!"
    exit 1
fi

docker run -id --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 20G -v /home/ubuntu/.cache:/root/.cache -v /home/ubuntu/docker/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} pt_inductor:$TAG

# Launch regular tests
if [ $TORCH_START_COMMIT == $TORCH_END_COMMIT ]; then
    docker cp /home/ubuntu/docker/inductor_test.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_train.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/version_collect.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_quant_performance.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_quant_accuracy.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_quant_acc.py $USER:/workspace/benchmark
    docker cp /home/ubuntu/docker/inductor_quant_acc_fp32.py $USER:/workspace/benchmark

    # Generate SW info out of real test
    docker exec -i $USER bash -c "bash version_collect.sh $LOG_DIR $DYNAMO_BENCH"

    prepare_imagenet(){
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
        mkdir -p /home/ubuntu/imagenet/val && mv ILSVRC2012_img_val.tar /home/ubuntu/imagenet/val && cd /home/ubuntu/imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
        wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
        bash valprep.sh
    }

    if [ $TEST_MODE == "inference" ]; then
        docker exec -i $USER bash -c "bash inductor_test.sh $THREADS $CHANNELS $PRECISION $TEST_SHAPE $LOG_DIR $WRAPPER $HF_TOKEN $BACKEND inference $SUITE $EXTRA"
    elif [ $TEST_MODE == "training_full" ]; then
        docker exec -i $USER bash -c "bash inductor_test.sh multiple $CHANNELS $PRECISION $TEST_SHAPE $LOG_DIR $WRAPPER $HF_TOKEN $BACKEND training $SUITE $EXTRA"
    elif [ $TEST_MODE == "training" ]; then
        docker exec -i $USER bash -c "bash inductor_train.sh $CHANNELS $PRECISION $LOG_DIR $EXTRA"
    elif [ $TEST_MODE == "performance" ]; then
        docker exec -i $USER bash -c "bash inductor_quant_performance.sh $LOG_DIR"
    elif [ $TEST_MODE == "accuracy" ]; then
        if [ ! -d "/home/ubuntu/imagenet" ];then
            prepare_imagenet
        fi
        docker cp /home/ubuntu/imagenet $USER:/workspace/benchmark/
        docker exec -i $USER bash -c "bash inductor_quant_accuracy.sh $LOG_DIR"
    elif [ $TEST_MODE == "all" ]; then
        if [ ! -d "/home/ubuntu/imagenet" ];then
            prepare_imagenet
        fi
        docker exec -i $USER bash -c "bash inductor_quant_performance.sh $LOG_DIR"
        docker cp /home/ubuntu/imagenet $USER:/workspace/benchmark/
        docker exec -i $USER bash -c "bash inductor_quant_accuracy.sh $LOG_DIR"
    fi
# Launch issue guilty commit search
else
    docker cp /home/ubuntu/docker/bisect_search.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/bisect_run_test.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_single_run.sh $USER:/workspace/pytorch
    # TODO: Hard code freeze on and default bs, add them as params future
    docker exec -i $USER bash -c "bash bisect_search.sh \
        START_COMMIT=$TORCH_START_COMMIT \
        END_COMMIT=$TORCH_END_COMMIT \
        SUITE=$SUITE \
        MODEL=$MODEL \
        MODE=$TEST_MODE \
        SCENARIO=$SCENARIO \
        PRECISION=$PRECISION \
        SHAPE=$TEST_SHAPE \
        WRAPPER=$WRAPPER \
        KIND=$KIND \
        THREADS=$THREADS \
        CHANNELS=$CHANNELS \
        FREEZE=on \
        BS=0 \
        LOG_DIR=$LOG_DIR \
        HF_TOKEN=$HF_TOKEN \
        BACKEND=$BACKEND \
        PERF_RATIO=$PERF_RATIO \
        EXTRA=$EXTRA" \
        > /home/ubuntu/docker/${LOG_DIR}/docker_exec_detailed.log
fi
