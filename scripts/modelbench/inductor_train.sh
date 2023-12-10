export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

# first / last
CHANNELS=${1:-first}
PRECISION=${2:-float32}
LOG_DIR=${3:-inductor_log}
<<<<<<< HEAD
DYNAMO_BENCH=${4:-fea73cb}
=======
EXTRA=${4}

>>>>>>> main
mkdir -p $LOG_DIR

# collect sw info
bash version_collect.sh ${LOG_DIR} ${DYNAMO_BENCH}

Channels_extra=""
if [[ ${CHANNELS} == "last" ]]; then
    Channels_extra="--channels-last "
fi

Flag_extra="$EXTRA "

torchbench="mobilenet_v2,resnet50"
huggingface="MobileBertForQuestionAnswering,Speech2Text2ForCausalLM,BlenderbotSmallForCausalLM,MobileBertForMaskedLM"
timm="fbnetv3_b,mobilenetv3_large_100,mnasnet_100,lcnet_050"

torchbench_model_list=($(echo "${torchbench}" | sed 's/,/ /g'))
huggingface_model_list=($(echo "${huggingface}" | sed 's/,/ /g'))
timm_model_list=($(echo "${timm}" | sed 's/,/ /g'))

CORES=$(lscpu | grep Core | awk '{print $4}')
export OMP_NUM_THREADS=$CORES
timestamp=$(date +%Y%m%d_%H%M%S)

for torchbench_model in ${torchbench_model_list[@]}; do
    # Commands for torchbench for device=cpu, dtype=float32 for training and for performance testing
    python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/torchbench.py --performance --${PRECISION} -dcpu --training --inductor --no-skip --dashboard --only ${torchbench_model} ${Channels_extra} ${Flag_extra} --cold_start_latency --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    # Commands for torchbench for device=cpu, dtype=float32 for training and for accuracy testing
    python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/torchbench.py --accuracy --${PRECISION} -dcpu --training --inductor --no-skip --dashboard --only ${torchbench_model} ${Channels_extra} ${Flag_extra} --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
done

for huggingface_model in ${huggingface_model_list[@]}; do
    if [[ ${huggingface_model} == "MobileBertForQuestionAnswering" ]]; then
        # Commands for huggingface for device=cpu, dtype=float32 for training and for performance testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --batch-size 32 --cold_start_latency --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        # Commands for huggingface for device=cpu, dtype=float32 for training and for accuracy testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    elif [[ ${huggingface_model} == "MobileBertForMaskedLM" ]]; then
        # Commands for huggingface for device=cpu, dtype=float32 for training and for performance testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --batch-size 64 --cold_start_latency --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        # Commands for huggingface for device=cpu, dtype=float32 for training and for accuracy testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    else
        # Commands for huggingface for device=cpu, dtype=float32 for training and for performance testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --cold_start_latency --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
        # Commands for huggingface for device=cpu, dtype=float32 for training and for accuracy testing
        python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --training --inductor --no-skip --dashboard --only ${huggingface_model} ${Channels_extra} ${Flag_extra} --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    fi
done

for timm_model in ${timm_model_list[@]}; do
    # Commands for timm_models for device=cpu, dtype=float32 for training and for performance testing
    python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/timm_models.py --performance --${PRECISION} -dcpu --training --inductor --no-skip --dashboard --only ${timm_model} ${Channels_extra} ${Flag_extra} --cold_start_latency --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_perf.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
    # Commands for timm_models for device=cpu, dtype=float32 for training and for accuracy testing
    python -m torch.backends.xeon.run_cpu --node_id 0 benchmarks/dynamo/timm_models.py --accuracy --${PRECISION} -dcpu --training --inductor --no-skip --dashboard --only ${timm_model} ${Channels_extra} ${Flag_extra} --output=${LOG_DIR}/multi_threads_channels_${CHANNELS}_${timestamp}_acc.csv 2>&1 | tee -a ${LOG_DIR}/multi_threads_model_bench_log_${timestamp}.log
done
