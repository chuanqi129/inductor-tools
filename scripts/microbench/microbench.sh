export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export USE_LLVM=False

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

LOG_DIR=${1:-dynamo_opbench}
suite=${2:-all}
repeats=${3:-30}
mkdir -p $LOG_DIR


timestamp=`date +%Y%m%d_%H%M%S`
if [ ${suite} == "all" ]; then
    numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite torchbench --op all --dtype float32 --repeats ${repeats} --device cpu 2>&1 | tee ${LOG_DIR}/multi_threads_opbench_torchbench_${timestamp}.log
    numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite huggingface --op all --dtype float32 --repeats ${repeats} --device cpu  2>&1 | tee ${LOG_DIR}/multi_threads_opbench_huggingface_${timestamp}.log
    numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite timm --op all --dtype float32 --repeats ${repeats} --device cpu  2>&1 | tee ${LOG_DIR}/multi_threads_opbench_timm_${timestamp}.log
else
    numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite ${suite} --op all --dtype float32 --repeats ${repeats} --device cpu  2>&1 | tee ${LOG_DIR}/multi_threads_opbench_${suite}_${timestamp}.log
fi
