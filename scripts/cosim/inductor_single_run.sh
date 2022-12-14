export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

SUITE=${1:-huggingface}
MODEL=${2:-GoogleFnet}

numactl -C 0-${end_core} --membind=0 python benchmarks/dynamo/${SUITE}.py --performance --float32 -dcpu -n50 --no-skip --dashboard --only "${MODEL}" --channels-last --backend=inductor
