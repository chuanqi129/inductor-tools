export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

precision=${1:-float32}
LOG_DIR=${2:-llm_bench}
jks_url=${3:-jks}
mkdir -p $LOG_DIR

# install transformers
pip uninstall transformers -y && pip install git+https://github.com/huggingface/transformers.git

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
python run_dynamo_gptj.py --use_dynamo --precision ${precision} 2>&1 | tee ${LOG_DIR}/llm_bench__${timestamp}.log
latency=$(grep "latency:" ${LOG_DIR}/llm_bench__${timestamp}.log | sed -e 's/.*latency//;s/[^0-9.]//')
echo latency : ${latency} >>${FILE}

# generate html report
cp generate_report.py ${curdir}/${LOG_DIR}
cd ${curdir}/${LOG_DIR}
python generate_report.py --url ${jks_url}
rm generate_report.py
