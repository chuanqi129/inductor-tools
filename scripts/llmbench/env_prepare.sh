set +e
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export TORCHINDUCTOR_FREEZING=1

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

precision=${1:-float32}
LOG_DIR=${2:-llm_bench}
mkdir -p $LOG_DIR

# kill unused process
itm_1=$(ps -ef | grep run_dynamo_llm.py | awk '{print $2}')

if [ -n "${itm_1:0}" ]; then
    sudo kill -9 ${itm_1:0}
    echo kill ${itm_1:0} successful
else
    echo not running ${itm_1:0}
fi

# install transformers
cd .. && rm -rf transformers && pip uninstall transformers -y
# release v4.31. + patch
git clone -b v4.31.0 https://github.com/huggingface/transformers.git
cd transformers && git apply ../token_latency.patch && python setup.py install && cd ..
cd /workspace/pytorch
# use offline mode for network issues
export TRANSFORMERS_OFFLINE=1

# collect sw info
curdir=$(pwd)
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


# result collect
function pre_collect() {
    key_word=$1
    log_file=$2
    res=$(grep "$key_word" "$log_file" | sed -e 's/.*latency: //;s/[^0-9.].*//')
    echo "$res"
}

function collect_perf() {
    log=$1
    num_s=$(grep 'Inference latency:' $log | sed -e 's/.*latency: //;s/[^0-9.].*//' | wc -l)
    latency=$(pre_collect 'Inference latency:' $log)
    first_latency=$(pre_collect 'First token average latency:' $log)
    avg_latency=$(pre_collect 'Average 2... latency:' $log)
    p90_latency=$(pre_collect 'P90 2... latency:' $log)
    p99_latency=$(pre_collect 'P99 2... latency:' $log)
    infer_latency=()   
    for i in $(seq $num_s); do
        r1=`echo $latency | awk '{print $'$i'}'`
        infer_latency[$i]=$r1
        r2=`echo $first_latency| awk '{print $'$i'}'`
        r3=`echo $avg_latency| awk '{print $'$i'}'`
        r4=`echo $p90_latency| awk '{print $'$i'}'`
        r5=`echo $p99_latency| awk '{print $'$i'}'`
        printf "$r1,$r2,$r3,$r4,$r5\\n" | tee -a ${FILE}
    done
    speedup1=`awk 'BEGIN{printf "%.2f",'${infer_latency[5]}' / '${infer_latency[1]}'}'`
    speedup2=`awk 'BEGIN{printf "%.2f",'${infer_latency[6]}' / '${infer_latency[2]}'}'`
    speedup3=`awk 'BEGIN{printf "%.2f",'${infer_latency[7]}' / '${infer_latency[3]}'}'`
    speedup4=`awk 'BEGIN{printf "%.2f",'${infer_latency[8]}' / '${infer_latency[4]}'}'`
    printf "$speedup1,$speedup2,$speedup3,$speedup4\\n" | tee -a ${FILE}
}


# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
# inductor
numactl -C 0-${end_core} --membind=0 python run_dynamo_llm.py --use_dynamo --token_latency --precision ${precision} 2>&1 | tee -a ${LOG_DIR}/llm_bench__${timestamp}.log
numactl -C 0-${end_core} --membind=0 python run_dynamo_llm.py --use_dynamo --token_latency --cpp_wrapper --precision ${precision} 2>&1 | tee -a ${LOG_DIR}/llm_bench__${timestamp}.log
# eager
numactl -C 0-${end_core} --membind=0 python run_dynamo_llm.py --token_latency --precision ${precision} 2>&1 | tee -a ${LOG_DIR}/llm_bench__${timestamp}.log
numactl -C 0-${end_core} --membind=0 python run_dynamo_llm.py --token_latency --cpp_wrapper --precision ${precision} 2>&1 | tee -a ${LOG_DIR}/llm_bench__${timestamp}.log
# collect metrics
collect_perf ${LOG_DIR}/llm_bench__${timestamp}.log
