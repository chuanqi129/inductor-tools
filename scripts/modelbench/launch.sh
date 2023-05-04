TAG=${1:-ww09.2}
TORCH_REPO=${2:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${3:-nightly}
TORCH_COMMIT=${4:nightly}

DYNAMO_BENCH=${5:-fea73cb}

AUDIO=${6:-0a652f5}
TEXT=${7:-c4ad5dd}
VISION=${8:-f2009ab}
DATA=${9:-5cb3e6d}
TORCH_BENCH=${10:-a0848e19}



echo "TAG" : $TAG
echo "TORCH_REPO" : $TORCH_REPO
echo "TORCH_BRANCH" : $TORCH_BRANCH
echo "TORCH_COMMIT" : $TORCH_COMMIT
echo "DYNAMO_BENCH" : $DYNAMO_BENCH
echo "AUDIO" : $AUDIO
echo "TEXT" : $TEXT
echo "VISION" : $VISION
echo "DATA" : $DATA
echo "TORCH_BENCH" : $TORCH_BENCH


# clean up
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -af

if [ -d inductor_log ]; then
    sudo rm -rf inductor_log
fi

DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=$TORCH_REPO --build-arg PT_BRANCH=$TORCH_BRANCH --build-arg PT_COMMIT=$TORCH_COMMIT --build-arg BENCH_COMMIT=$DYNAMO_BENCH --build-arg TORCH_AUDIO_COMMIT=$AUDIO --build-arg TORCH_TEXT_COMMIT=$TEXT --build-arg TORCH_VISION_COMMIT=$VISION --build-arg TORCH_DATA_COMMIT=$DATA --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH --build-arg https_proxy=${https_proxy} -t pt_inductor:$TAG -f Dockerfile --target image .

docker run -id --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/ubuntu/docker/download/hub/checkpoints:/root/.cache/torch/hub/checkpoints -v /home/ubuntu/docker/inductor_log:/workspace/pytorch/inductor_log pt_inductor:$TAG


docker exec -i $USER bash -c "bash inductor_test.sh all first float32 static inductor_log $DYNAMO_BENCH"
