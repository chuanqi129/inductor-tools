export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

LOG_DIR=${1:-dynamo_ww48_5}
mkdir -p $LOG_DIR

# workaround for microbenchmark
sed -i "s;raise e;#raise e;"  benchmarks/dynamo/microbenchmarks/operatorbench.py

timestamp=`date +%Y%m%d_%H%M%S`
numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite torchbench --op all --dtype float32 --repeats 30 --device cpu 2>&1 | tee ${LOG_DIR}/multi_threads_opbench_torchbench_${timestamp}.log
numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite huggingface --op all --dtype float32 --repeats 30 --device cpu  2>&1 | tee ${LOG_DIR}/multi_threads_opbench_huggingface_${timestamp}.log
numactl -C 0-$end_core --membind 0 python benchmarks/dynamo/microbenchmarks/operatorbench.py --suite timm --op all --dtype float32 --repeats 30 --device cpu  2>&1 | tee ${LOG_DIR}/multi_threads_opbench_timm_${timestamp}.log