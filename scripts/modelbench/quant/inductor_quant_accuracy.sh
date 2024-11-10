export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
mkdir inductor_quant_acc/
pip install networkx==2.8
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

numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python inductor_quant_acc.py 2>&1 |& tee "./inductor_quant_acc_ptq.log"
mv ./inductor_quant_acc_ptq.log inductor_quant_acc/
rm -rf /tmp/*
TORCHINDUCTOR_CPP_WRAPPER=1 numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python inductor_quant_acc.py --cpp_wrapper 2>&1 |& tee "./inductor_quant_acc_ptq_cpp_wrapper.log"
mv ./inductor_quant_acc_ptq_cpp_wrapper.log inductor_quant_acc/
rm -rf /tmp/*
numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python inductor_quant_acc.py  --is_qat 2>&1 |& tee "./inductor_quant_acc_qat.log"
mv ./inductor_quant_acc_qat.log inductor_quant_acc/
numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} python inductor_quant_acc.py --is_fp32 2>&1 |& tee "./inductor_quant_acc_fp32.log"
mv ./inductor_quant_acc_fp32.log inductor_quant_acc/

mv inductor_quant_acc ../pytorch/$LOG_DIR/
