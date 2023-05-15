set +e
TAG=${1:-ww18.4}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
SHAPE=${4:-static}

TORCH_REPO=${5:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${6:-nightly}
TORCH_COMMIT=${7:nightly}

DYNAMO_BENCH=${8:-fea73cb}

IPEX_REPO=${9:-https://github.com/intel/intel-extension-for-pytorch.git}
IPEX_BRANCH=${10:-master}
IPEX_COMMIT=${11:master}


# kill unused process
itm_1=$(ps -ef | grep entrance.sh | awk '{print $2}')
itm_2=$(ps -ef | grep launch.sh | awk '{print $2}')
itm_3=$(ps -ef | grep inductor_test.sh | awk '{print $2}')
itm_4=$(ps -ef | grep inductor_train.sh | awk '{print $2}')

if [ -n "${itm_1}" ]; then
    sudo kill -9 $item_1
fi

if [ -n "${itm_2}" ]; then
    sudo kill -9 $item_2
fi

if [ -n "${itm_3}" ]; then
    sudo kill -9 $item_3
fi

if [ -n "${itm_4}" ]; then
    sudo kill -9 $item_4
fi

# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

# rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt file
if [ -f finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt ]; then
    rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi

# launch benchmark
bash launch.sh ${TAG} ${PRECISION} ${TEST_MODE} ${TORCH_REPO} ${TORCH_BRANCH} ${TORCH_COMMIT} ${DYNAMO_BENCH} ${IPEX_REPO} ${IPEX_BRANCH} ${IPEX_COMMIT}

# create finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    touch finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi
