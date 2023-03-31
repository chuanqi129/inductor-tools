export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

transformers_version=${1:-4.24.0}
precision=${2:-float32}
LOG_DIR=${3:-llm_bench}
mkdir -p $LOG_DIR

# install transformers
pip uninstall transformers -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install transformers==${transformers_version}

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
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__)''' >>${FILE}
echo transformers : ${transformers_version} >>${FILE}
echo precision : ${precision} >>${FILE}

# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
python run_dynamo_gptj.py --use_dynamo --precision ${precision} --greedy 2>&1 | tee ${LOG_DIR}/llm_bench__${timestamp}.log
latency=$(grep "latency:" ${LOG_DIR}/llm_bench__${timestamp}.log | sed -e 's/.*latency//;s/[^0-9.]//')
echo latency : ${latency} >>${FILE}

# generate html report
cp generate_report.py ${curdir}/${LOG_DIR}
cd ${curdir}/${LOG_DIR}
python generate_report.py
