LOG_DIR=${1:-inductor_log}
DYNAMO_BENCH=${2:-fea73cb}

# collect sw info
curdir=$(pwd)
mkdir -p ${curdir}/${LOG_DIR}
touch ${curdir}/${LOG_DIR}/version.txt
cd /workspace/benchmark
echo torchbench : $(git rev-parse --short HEAD) >>${curdir}/${LOG_DIR}/version.txt
echo "oneDNN_commit : " $(git ls-remote -- https://github.com/oneapi-src/oneDNN.git | grep "refs/heads/main" |awk '{print $1}') >>${curdir}/${LOG_DIR}/version.txt
cd /workspace/pytorch
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__)''' >>${curdir}/${LOG_DIR}/version.txt
echo dynamo_benchmarks : $DYNAMO_BENCH >>${curdir}/${LOG_DIR}/version.txt

