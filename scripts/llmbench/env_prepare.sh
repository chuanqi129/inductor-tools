export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

precision=${1:-float32}
LOG_DIR=${2:-llm_bench}
mkdir -p $LOG_DIR

# install transformers
pip uninstall transformers -y && pip install transformers

# use offline mode for network issues
export TRANSFORMERS_OFFLINE=1

# collect sw info
curdir=`pwd`
FILE=${curdir}/${LOG_DIR}/result.txt
if [ -f ${FILE} ]; then
    rm ${FILE}
fi
touch ${FILE}

cd /workspace/benchmark
echo torchbench : $(git rev-parse --short HEAD) >>${FILE}
cd /workspace/pytorch
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata,transformers; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__); \
        print("transformers : ", transformers.__version__)''' >>${FILE}
echo precision : ${precision} >>${FILE}

# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
numactl -C 0-${end_core} --membind=0 python run_dynamo_llm.py --use_dynamo --precision ${precision} 2>&1 | tee ${LOG_DIR}/llm_bench__${timestamp}.log
latency=$(grep "latency:" ${LOG_DIR}/llm_bench__${timestamp}.log | sed -e 's/.*latency//;s/[^0-9.]//')
echo latency : ${latency} >>${FILE}