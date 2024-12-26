export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
models=alexnet,shufflenet_v2_x1_0,mobilenet_v3_large,vgg16,densenet121,mnasnet1_0,squeezenet1_1,mobilenet_v2,resnet50,resnet152,resnet18,resnext50_32x4d
#models=alexnet,shufflenet_v2_x1_0,mobilenet_v3_large,vgg16,mnasnet1_0,squeezenet1_1,mobilenet_v2,resnet50,resnet152,resnet18,resnext50_32x4d
#models=alexnet
rm -rf .userbenchmark
mkdir inductor_quant/
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+schedule,+inductor,+output_code"
cpu_allowed_list=$(cat /proc/self/status | grep Cpus_allowed_list | awk '{print $2}')
start_core=$(echo ${cpu_allowed_list} | awk -F- '{print $1}')
mem_allowed_list=$(cat /proc/self/status | grep Mems_allowed_list | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep Core | awk '{print $4}')
NUM_SOCKET=$(lscpu | grep "Socket(s)" | awk '{print $2}')
NUM_NUMA=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
CORES=$(expr $CORES_PER_SOCKET \* $NUM_SOCKET / $NUM_NUMA)
end_core=$(expr ${start_core} + ${CORES} - 1)
cpu_allowed_list="${start_core}-${end_core}"
if [[ ${mem_allowed_list} =~ '-' ]];then
    mem_allowed_list=$(echo ${mem_allowed_list} | awk -F- '{print $1}')
fi
 
TORCHINDUCTOR_FREEZING=1 numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/ptq
TORCHINDUCTOR_FREEZING=1 TORCHINDUCTOR_CPP_WRAPPER=1 numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize --cpp_wrapper -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/cpp
TORCHINDUCTOR_FREEZING=1 numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize --is_qat -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/qat
TORCHINDUCTOR_FREEZING=1 numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python run_benchmark.py cpu -m ${models} --torchdynamo inductor -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/general_inductor

mv inductor_quant ../pytorch/$LOG_DIR/
