set -x

export TORCHINDUCTOR_FREEZING=1

export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so
dtype=${1:-bf16}
iter=${2:-5}
cd /workspace/torchtune/
if [ ! -d "torchtune_log" ]; then
    mkdir -p "torchtune_log"
fi
#meta-llama/Llama-3.2-1B-Instruct KD
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
tune run knowledge_distillation_single_device --config llama3_2/8B_to_1B_KD_lora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Llama-3.2-1B-Instruct_KD.log
#mistralai/Mistral-7B-Instruct-v0.2 ppo
tune download weqweasdas/RM-Mistral-7B --output-dir /tmp/RM-Mistral-7B/
tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir /tmp/Mistral-7B-Instruct-v0.2/ 
tune run ppo_full_finetune_single_device --config mistral/7B_full_ppo_low_memory device=xpu dtype=$dtype optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 seed=123 2>&1 | tee torchtune_log/Mistral-7B-Instruct-v0.2_ppo.log
#meta-llama/Meta-Llama-3.1-8B-Instruct lora_dpo
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
tune run lora_dpo_single_device --config llama3_1/8B_lora_dpo_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_lora_dpo.log
#meta-llama/Meta-Llama-3.1-8B-Instruct full finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
tune run full_finetune_single_device --config llama3_1/8B_full_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 seed=123 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_full.log
#mistralai/Mistral-7B-v0.1 full finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
tune run full_finetune_single_device --config mistral/7B_full_low_memory device=xpu dtype=$dtype max_steps_per_epoch=$iter optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 enable_activation_offloading=False seed=123 2>&1 | tee torchtune_log/Mistral-7B-v0.1_full.log
#meta-llama/Meta-Llama-3.1-8B-Instruct lora finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_lora.log
#mistralai/Mistral-7B-v0.1 lora finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
tune run lora_finetune_single_device --config mistral/7B_lora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Mistral-7B-v0.1_lora.log
#meta-llama/Meta-Llama-3.1-8B-Instruct qlora finetune
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3.1-8B-Instruct_qlora.log
#mistralai/Mistral-7B-v0.1 qlora finetune
tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1
tune run lora_finetune_single_device --config mistral/7B_qlora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Mistral-7B-v0.1_qlora.log
#meta-llama/Meta-Llama-3-8B-Instruct dora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_dora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_dora.log
#meta-llama/Meta-Llama-3-8B-Instruct qdora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_qdora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qdora.log
#meta-llama/Meta-Llama-3-8B-Instruct qlora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_qlora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qlora.log
#meta-llama/Meta-Llama-3-8B-Instruct lora
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run lora_finetune_single_device --config llama3/8B_lora_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_lora.log
#meta-llama/Meta-Llama-3-8B-Instruct full
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct
tune run full_finetune_single_device --config llama3/8B_full_single_device device=xpu dtype=$dtype max_steps_per_epoch=$iter optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 seed=123 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_full.log
if [[ $dtype == 'bf16' ]]; then
    #meta-llama/Meta-Llama-3.1-8B-Instruct qlora finetune 300 step
    tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=xpu dtype=$dtype max_steps_per_epoch=300 2>&1 | tee torchtune_log/Meta-Llama-3-8B-Instruct_qlora_300.log
fi
