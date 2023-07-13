set -x
TAG=${1:-ww09.2}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
TEST_SHAPE=${4:-static}

TORCH_REPO=${5:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${6:-nightly}
TORCH_COMMIT=${7:-9a8c655}

DYNAMO_BENCH=${8:-1238ae3}

IPEX_REPO=${9:-https://github.com/intel/intel-extension-for-pytorch.git}
IPEX_BRANCH=${10:-master}
IPEX_COMMIT=${11:-master}

AUDIO=${12:-d9643f5}
TEXT=${13:-b0ebddc}
VISION=${14:-9b7c7d3}
DATA=${15:-65e2ede}
TORCH_BENCH=${16:-ec359fad}
THREAD=${17:-all}



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
echo "IPEX_REPO" : $IPEX_REPO
echo "IPEX_BRANCH" : $IPEX_BRANCH
echo "IPEX_COMMIT" : $IPEX_COMMIT

# clean up
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -af

if [ -d ipex_log ]; then
    sudo rm -rf ipex_log
fi

docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=$TORCH_REPO --build-arg PT_BRANCH=$TORCH_BRANCH --build-arg PT_COMMIT=$TORCH_COMMIT --build-arg IPEX_REPO=$IPEX_REPO --build-arg IPEX_BRANCH=$IPEX_BRANCH --build-arg IPEX_COMMIT=$IPEX_COMMIT --build-arg BENCH_COMMIT=$DYNAMO_BENCH --build-arg TORCH_AUDIO_COMMIT=$AUDIO --build-arg TORCH_TEXT_COMMIT=$TEXT --build-arg TORCH_VISION_COMMIT=$VISION --build-arg TORCH_DATA_COMMIT=$DATA --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH --build-arg https_proxy=${https_proxy} -t ipex_torchbench:$TAG -f Dockerfile.ipex --target image .

docker run -id --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 1G -v /home/ubuntu/docker/download/hub/checkpoints:/root/.cache/torch/hub/checkpoints -v /home/ubuntu/docker/ipex_log:/workspace/pytorch/ipex_log ipex_torchbench:$TAG

docker cp /home/ubuntu/docker/ipex_test.sh $USER:/workspace/pytorch

if (($TEST_MODE == "inference")); then
    docker exec -i $USER bash -c "bash ipex_test.sh ${THREAD} first $PRECISION $TEST_SHAPE ipex_log $DYNAMO_BENCH"
elif (($TEST_MODE == "training")); then
    docker exec -i $USER bash -c "bash inductor_train.sh first $PRECISION inductor_log $DYNAMO_BENCH"
fi
