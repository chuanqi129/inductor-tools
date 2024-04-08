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
HF_TOKEN=${17:-hf_xx}
BACKEND=${18:-inductor}
SUITE=${19:-all}
MODEL=${20:-resnet50}
TORCH_START_COMMIT=${21:-${TORCH_BRANCH}}
TORCH_END_COMMIT=${22:-${TORCH_START_COMMIT}}
SCENARIO=${23:-accuracy}
KIND=${24:-crash} # issue kind crash/drop/fixed/improve
PERF_RATIO=${25:-1.1}
IPEX_REPO=${26:-https://github.com/intel/intel-extension-for-pytorch.git}
IPEX_BRANCH=${27:-main}
IPEX_COMMIT=${28:-${IPEX_BRANCH}}

EXTRA=${29}



# cd target dir
echo cur_dir :$(pwd)
cd /home/ubuntu/docker

# rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt file
if [ -f finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt ]; then
    rm finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi

# launch benchmark
bash launch.sh ${TAG} ${PRECISION} ${TEST_MODE} ${SHAPE} ${TORCH_REPO} ${TORCH_BRANCH} ${TORCH_COMMIT} ${DYNAMO_BENCH} ${AUDIO} ${TEXT} ${VISION} ${DATA} ${TORCH_BENCH} ${THREADS} ${CHANNELS} ${WRAPPER} ${HF_TOKEN} ${BACKEND} ${SUITE} ${MODEL} ${TORCH_START_COMMIT} ${TORCH_END_COMMIT} ${SCENARIO} ${KIND} ${PERF_RATIO} ${IPEX_REPO} ${IPEX_BRANCH} ${IPEX_COMMIT} ${EXTRA} 

# create finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    echo "Finished!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
else
    echo "benchmark failed!"
    echo "Failed!" > finished_${PRECISION}_${TEST_MODE}_${SHAPE}.txt
fi
