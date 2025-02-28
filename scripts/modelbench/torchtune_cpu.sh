set -x

export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911

cpu_allowed_list=$(cat /proc/self/status | grep Cpus_allowed_list | awk '{print $2}')
start_core=$(echo ${cpu_allowed_list} | awk -F- '{print $1}')
mem_allowed_list=$(cat /proc/self/status | grep Mems_allowed_list | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep Core | awk '{print $4}')
NUM_SOCKET=$(lscpu | grep "Socket(s)" | awk '{print $2}')
NUM_NUMA=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
CORES=$(expr $CORES_PER_SOCKET \* $NUM_SOCKET / $NUM_NUMA)
if [[ ${mem_allowed_list} =~ '-' ]];then
    end_core=$(expr ${start_core} + ${CORES} - 1)
    cpu_allowed_list="${start_core}-${end_core}"
    mem_allowed_list=$(echo ${mem_allowed_list} | awk -F- '{print $1}')
fi

export OMP_NUM_THREADS=$CORES
end_core=$(expr $CORES - 1)
export TORCHINDUCTOR_FREEZING=1

export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so
dtype=${1:-bf16}
iter=${2:-5}
cd /workspace/torchtune/
if [ ! -d "torchtune_log" ]; then
    mkdir -p "torchtune_log"
fi
#meta-llama/Llama-3.2-1B-Instruct KD
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run knowledge_distillation_single_device --config llama3_2/8B_to_1B_KD_lora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Llama-3.2-1B-Instruct_KD.log
#mistralai/Mistral-7B-Instruct-v0.2 ppo
tune download weqweasdas/RM-Mistral-7B --output-dir /tmp/RM-Mistral-7B/
tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir /tmp/Mistral-7B-Instruct-v0.2/ 
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run ppo_full_finetune_single_device --config mistral/7B_full_ppo_low_memory device=cpu dtype=$dtype optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 2>&1 | tee torchtune_log/Mistral-7B-Instruct-v0.2_ppo.log
#meta-llama/Meta-Llama-3.1-8B-Instruct lora_dpo
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_dpo_single_device --config llama3_1/8B_lora_dpo_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_lora_dpo.log
#meta-llama/Meta-Llama-3.1-8B-Instruct full finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run full_finetune_single_device --config llama3_1/8B_full_single_device device=cpu dtype=$dtype optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_full.log
#mistralai/Mistral-7B-v0.1 full finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run full_finetune_single_device --config mistral/7B_full_low_memory device=cpu dtype=$dtype optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 enable_activation_offloading=False max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Mistral-7B-v0.1_full.log
#meta-llama/Meta-Llama-3.1-8B-Instruct lora finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_lora.log
#mistralai/Mistral-7B-v0.1 lora finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config mistral/7B_lora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Mistral-7B-v0.1_lora.log
#meta-llama/Meta-Llama-3.1-8B-Instruct qlora finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_qlora.log
#mistralai/Mistral-7B-v0.1 qlora finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config mistral/7B_qlora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Mistral-7B-v0.1_qlora.log
#meta-llama/Meta-Llama-3-8B-Instruct dora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config llama3/8B_dora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_dora.log
#meta-llama/Meta-Llama-3-8B-Instruct qdora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config llama3/8B_qdora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qdora.log
#meta-llama/Meta-Llama-3-8B-Instruct qlora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_qlora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qlora.log
#meta-llama/Meta-Llama-3-8B-Instruct lora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_lora_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_lora.log
#meta-llama/Meta-Llama-3-8B-Instruct full
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_full_single_device device=cpu dtype=$dtype max_steps_per_epoch=$iter optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_full.log
if [[ $dtype == 'bf16' ]]; then
    #meta-llama/Meta-Llama-3.1-8B-Instruct qlora finetune 300 step
    numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=cpu dtype=$dtype max_steps_per_epoch=300 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qlora_300.log
fi
