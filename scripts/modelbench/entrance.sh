set +e
TAG=${1:-ww18.4}
PRECISION=${2:-float32}
TEST_MODE=${3:-inference}
SHAPE=${4:-static}

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
# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

# rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt file
if [ -f finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt ]; then
    rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi

# launch benchmark
bash launch.sh ${TAG} ${PRECISION} ${TEST_MODE} ${SHAPE} ${TORCH_REPO} ${TORCH_BRANCH} ${TORCH_COMMIT} ${ONEDNN_BRANCH} ${DYNAMO_BENCH} ${AUDIO} ${TEXT} ${VISION} ${DATA} ${TORCH_BENCH} ${THREADS} ${CHANNELS} ${WRAPPER} ${HF_TOKEN} ${BACKEND} ${SUITE} ${MODEL} ${TORCH_START_COMMIT} ${TORCH_END_COMMIT} ${SCENARIO} ${KIND} ${PERF_RATIO} ${EXTRA}

# create finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    echo "Finished!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
else
    echo "benchmark failed!"
    echo "Failed!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi
