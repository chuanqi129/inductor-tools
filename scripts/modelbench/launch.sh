TAG=${1:-ww09.2}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
TEST_SHAPE=${4:-static}

TORCH_REPO=${5:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${6:-nightly}
TORCH_COMMIT=${7:-nightly}
ONEDNN_BRANCH=${8:-default}

DYNAMO_BENCH=${9:-fea73cb}

AUDIO=${10:-0a652f5}
TEXT=${11:-c4ad5dd}
VISION=${12:-f2009ab}
DATA=${13:-5cb3e6d}
TORCH_BENCH=${14:-a0848e19}

THREADS=${15:-all}
CHANNELS=${16:-first}
WRAPPER=${17:-default}
HF_TOKEN=${18:-hf_xx}
BACKEND=${19:-inductor}
SUITE=${20:-all}
MODEL=${21:-resnet50}
TORCH_START_COMMIT=${22:-${TORCH_BRANCH}}
TORCH_END_COMMIT=${23:-${TORCH_START_COMMIT}}
SCENARIO=${24:-accuracy}
KIND=${25:-crash} # issue kind crash/drop
PERF_RATIO=${26:-1.1}
EXTRA=${27}

echo "TAG" : $TAG
echo "PRECISION" : $PRECISION
echo "TEST_MODE" : $TEST_MODE
echo "TEST_SHAPE" : $TEST_SHAPE
echo "TORCH_REPO" : $TORCH_REPO
echo "TORCH_BRANCH" : $TORCH_BRANCH
echo "TORCH_COMMIT" : $TORCH_COMMIT
echo "DYNAMO_BENCH" : $DYNAMO_BENCH
echo "ONEDNN_BRANCH" : $ONEDNN_BRANCH
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
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -af

LOG_DIR="inductor_log"
if [ -d ${LOG_DIR} ]; then
    sudo rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}

DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=$TORCH_REPO --build-arg ONEDNN_BRANCH=$ONEDNN_BRANCH --build-arg PT_BRANCH=$TORCH_BRANCH --build-arg PT_COMMIT=$TORCH_COMMIT --build-arg BENCH_COMMIT=$DYNAMO_BENCH --build-arg TORCH_AUDIO_COMMIT=$AUDIO --build-arg TORCH_TEXT_COMMIT=$TEXT --build-arg TORCH_VISION_COMMIT=$VISION --build-arg TORCH_DATA_COMMIT=$DATA --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH --build-arg https_proxy=${https_proxy} --build-arg HF_HUB_TOKEN=$HF_TOKEN -t pt_inductor:$TAG -f Dockerfile --target image . > ${LOG_DIR}/image_build.log 2>&1
# Early exit for docker image build issue
image_status=`tail -n 5 ${LOG_DIR}/image_build.log | grep ${TAG} | wc -l`
if [ $image_status -eq 0 ]; then
    echo "Docker image build filed, early exit!"
    exit 1
fi

docker run -id --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 20G -v /home/ubuntu/.cache:/root/.cache -v /home/ubuntu/docker/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} pt_inductor:$TAG

# Launch regular tests
if [ $TORCH_START_COMMIT == $TORCH_END_COMMIT ]; then
    docker cp /home/ubuntu/docker/inductor_test.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_train.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/version_collect.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/cpu_usebm.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/cpu_usebm_train.sh $USER:/workspace/pytorch

    # Generate SW info out of real test
    docker exec -i $USER bash -c "bash version_collect.sh $LOG_DIR $DYNAMO_BENCH"

    if [ $TEST_MODE == "inference" ]; then
        docker exec -i $USER bash -c "bash inductor_test.sh $THREADS $CHANNELS $PRECISION $TEST_SHAPE $LOG_DIR $WRAPPER $HF_TOKEN $BACKEND inference $SUITE $EXTRA"
    elif [ $TEST_MODE == "training_full" ]; then
        docker exec -i $USER bash -c "bash inductor_test.sh multiple $CHANNELS $PRECISION $TEST_SHAPE $LOG_DIR $WRAPPER $HF_TOKEN $BACKEND training $SUITE $EXTRA"
    elif [ $TEST_MODE == "training" ]; then
        docker exec -i $USER bash -c "bash inductor_train.sh $CHANNELS $PRECISION $LOG_DIR $EXTRA"
    elif [ $TEST_MODE == "user_benchmark_infer" ]; then
        docker exec -i $USER bash -c "bash cpu_usebm.sh"
    elif [ $TEST_MODE == "user_benchmark_train" ]; then
        docker exec -i $USER bash -c "bash cpu_usebm_train.sh"
    elif [ $TEST_MODE == "user_benchmark" ]; then
        docker exec -i $USER bash -c "bash cpu_usebm.sh"
    fi
# Launch issue guilty commit search
else
    docker cp /home/ubuntu/docker/bisect_search.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/bisect_run_test.sh $USER:/workspace/pytorch
    docker cp /home/ubuntu/docker/inductor_single_run.sh $USER:/workspace/pytorch
    # TODO: Hard code freeze on and default bs, add them as params future
    docker exec -i $USER bash -c "bash bisect_search.sh $TORCH_BRANCH $TORCH_START_COMMIT $TORCH_END_COMMIT $SUITE $MODEL $TEST_MODE $SCENARIO $PRECISION $TEST_SHAPE $WRAPPER $KIND $THREADS $CHANNELS on 0 $LOG_DIR $HF_TOKEN $BACKEND $PERF_RATIO $EXTRA"
fi
