export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

# collect sw info
curdir=`pwd`
if [ -f "version.txt" ]; then
rm version.txt
fi

touch ${curdir}/version.txt
cd /workspace/benchmark
echo torchbench : `git rev-parse --short HEAD` >> ${curdir}/version.txt
cd /workspace/pytorch
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__)''' >> ${curdir}/version.txt