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
cd ${LOG_DIR}
if [ -f "result.txt" ]; then
    rm result.txt
fi
touch result.txt

cd /workspace/benchmark
echo torchbench : $(git rev-parse --short HEAD) >>${LOG_DIR}/result.txt
cd /workspace/pytorch
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__)''' >>${LOG_DIR}/result.txt
echo transformers : ${transformers_version} >>${LOG_DIR}/result.txt
echo precision : ${precision} >>${LOG_DIR}/result.txt

# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
python run_dynamo_gptj.py --use_dynamo --precision ${precision} --greedy 2>&1 | tee ${LOG_DIR}/llm_bench__${timestamp}.log
latency=$(grep "Inference latency :" ${LOG_DIR}/llm_bench__${timestamp}.log | sed -e 's/.*latency//;s/[^0-9.]//g')
echo latency : ${latency} >>${LOG_DIR}/result.txt

# generate html report
python generate_report.py
cp llm_report.html ${LOG_DIR}