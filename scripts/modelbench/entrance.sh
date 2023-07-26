set +e
TAG=${1:-ww18.4}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
SHAPE=${4:-static}

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

# kill unused process
itm_1=$(ps -ef | grep entrance | awk '{print $2}')
itm_2=$(ps -ef | grep launch | awk '{print $2}')
itm_3=$(ps -ef | grep inductor | awk '{print $2}')

if [ -n "${itm_1:0}" ]; then
    sudo kill -9 ${itm_1:0}
    echo kill ${itm_1:0} successful
else
    echo not running ${itm_1:0}
fi

if [ -n "${itm_2:0}" ]; then
    sudo kill -9 ${itm_2:0}
    echo kill ${itm_2:0} successful
else
    echo not running ${itm_2:0}
fi

if [ -n "${itm_3:0}" ]; then
    sudo kill -9 ${itm_3:0}
    echo kill ${itm_3:0} successful
else
    echo not running ${itm_3:0}
fi

# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

# rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt file
if [ -f finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt ]; then
    rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi

# launch benchmark
bash launch.sh ${TAG} ${PRECISION} ${TEST_MODE} ${SHAPE} ${TORCH_REPO} ${TORCH_BRANCH} ${TORCH_COMMIT} ${DYNAMO_BENCH} ${AUDIO} ${TEXT} ${VISION} ${DATA} ${TORCH_BENCH} ${THREADS} ${CHANNELS} ${WRAPPER}

# create finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    touch finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi
