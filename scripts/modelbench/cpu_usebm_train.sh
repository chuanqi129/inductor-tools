export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
mkdir userbenchmark_aws/

echo running cpu userbenchmark........
cmd_prefix='''python run_benchmark.py cpu --test train --channels-last --launcher --launcher-args="--throughput-mode" --metrics throughputs'''

# FP32 eager
${cmd_prefix}
mv .userbenchmark/cpu eager_throughtput_fp32_train
mv eager_throughtput_fp32_train userbenchmark_aws/

# BF16 eager
${cmd_prefix} --precision amp_bf16
mv .userbenchmark/cpu eager_throughtput_bf16_train
mv eager_throughtput_bf16_train userbenchmark_aws/

mv userbenchmark_aws ../pytorch/$LOG_DIR/

