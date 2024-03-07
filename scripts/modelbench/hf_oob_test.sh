set +e

source /home/ubuntu/docker/env_groovy.txt

# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

if [ -f finished.txt ]; then
    rm finished.txt
fi

# clean up
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -af

if [ -d ${LOG_DIR} ]; then
    sudo rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}

docker build \
        --build-arg TORCH_REPO=$TORCH_REPO \
        --build-arg TORCH_BRANCH=$TORCH_BRANCH \
        --build-arg TORCH_COMMIT=$TORCH_COMMIT \
        --build-arg TORCH_VISION_COMMIT=$VISION \
        --build-arg TRANSFORMERS_VERSION=$TRANSFORMERS_VERSION \
        --build-arg HF_TEST_REPO=$HF_TEST_REPO \
        --build-arg HF_TEST_BRANCH=$HF_TEST_BRANCH \
        --build-arg HF_TEST_COMMIT=$HF_TEST_COMMIT \
        --build-arg http_proxy=${http_proxy} \
        --build-arg https_proxy=${https_proxy} \
        -t hf_oob_test:${target} -f Dockerfile.hf_oob --target image . > ${LOG_DIR}/image_build.log 2>&1

# Early exit for docker image build issue
image_status=`tail -n 5 ${LOG_DIR}/image_build.log | grep ${target} | wc -l`
if [ $image_status -eq 0 ]; then
    echo "Docker image build filed, early exit!"
    exit 1
fi

docker run -id --name $USER --privileged \
        --env https_proxy=${https_proxy} \
        --env http_proxy=${http_proxy} \
        --env LOG_DIR=${LOG_DIR} \
        --net host --shm-size 20G \
        -v /home/ubuntu/.cache:/root/.cache \
        -v /home/ubuntu/docker/${LOG_DIR}:/workspace/hf_testcase/${LOG_DIR} hf_oob_test:${target}

docker exec -e CONDA_PREFIX='/opt/conda' -i $USER bash -c "bash run_cpu.sh"
status=${PIPESTATUS[0]}
# create finished.txt when finished
if [ ${status} -eq 0 ]; then
    echo "benchmark finished!"
    echo "Finished!" > finished.txt
else
    echo "benchmark failed!"
    echo "Failed!" > finished.txt
fi