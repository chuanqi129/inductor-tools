TAG=${1:-ww09.2}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
TEST_SHAPE=${4:-static}

TORCH_REPO=${5:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${6:-nightly}
TORCH_COMMIT=${7:-nightly}

DYNAMO_BENCH=${8:-fea73cb}

AUDIO=${9:-0a652f5}
TEXT=${10:-c4ad5dd}
VISION=${11:-f2009ab}
DATA=${12:-5cb3e6d}
TORCH_BENCH=${13:-a0848e19}

THREADS=${14:-all}
CHANNELS=${15:-first}
WRAPPER=${16:-default}
HF_TOKEN=${17:-hf_xx}

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

# clean up
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -af

if [ -d inductor_log ]; then
    sudo rm -rf inductor_log
fi

DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=$TORCH_REPO --build-arg PT_BRANCH=$TORCH_BRANCH --build-arg PT_COMMIT=$TORCH_COMMIT --build-arg BENCH_COMMIT=$DYNAMO_BENCH --build-arg TORCH_AUDIO_COMMIT=$AUDIO --build-arg TORCH_TEXT_COMMIT=$TEXT --build-arg TORCH_VISION_COMMIT=$VISION --build-arg TORCH_DATA_COMMIT=$DATA --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH --build-arg https_proxy=${https_proxy} -t pt_inductor:$TAG -f Dockerfile --target image .

docker run -id --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 20G -v /home/ubuntu/docker/download/hub/checkpoints:/root/.cache/torch/hub/checkpoints -v /home/ubuntu/docker/inductor_log:/workspace/pytorch/inductor_log pt_inductor:$TAG

docker cp /home/ubuntu/docker/inductor_test.sh $USER:/workspace/pytorch
docker cp /home/ubuntu/docker/inductor_train.sh $USER:/workspace/pytorch
docker cp /home/ubuntu/docker/version_collect.sh $USER:/workspace/pytorch

if [ $TEST_MODE == "inference" ]; then
    docker exec -i $USER bash -c "bash inductor_test.sh $THREADS $CHANNELS $PRECISION $TEST_SHAPE inductor_log $DYNAMO_BENCH $WRAPPER $HF_TOKEN"
elif [ $TEST_MODE == "training" ]; then
    docker exec -i $USER bash -c "bash inductor_train.sh $CHANNELS $PRECISION inductor_log $DYNAMO_BENCH"
fi
