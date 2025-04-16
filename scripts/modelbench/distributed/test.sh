#FSDP
git clone -b release/xpu/2.6.10 https://github.com/intel/intel-extension-for-pytorch.git ipex
cd ipex/examples/gpu/llm/fine-tuning
pip install -r requirements.txt
cd Llama3
git clone https://github.com/huggingface/accelerate.git
cd accelerate
#change self.backend = "xccl" in https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L198
pip install -e .
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1
export model="meta-llama/Meta-Llama-3-8B"
#full fine-tuning
accelerate launch --config_file "fsdp_config.yaml" llama3_ft.py --model_name_or_path ${model} --use_flashattn False --bf16 True --max_seq_length 128 --output_dir="output" --evaluation_strategy="epoch" --learning_rate=1e-3 --gradient_accumulation_steps=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs=1 --save_steps=500 --logging_steps=1 --save_total_limit=8
#LoRA fine-tuning
accelerate launch --config_file "fsdp_config.yaml" llama3_ft.py --model_name_or_path ${model} --use_flashattn False --bf16 True --use_peft True --max_seq_length 128 --output_dir="output" --evaluation_strategy="epoch" --learning_rate=1e-3 --gradient_accumulation_steps=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs=1 --save_steps=500 --logging_steps=1 --save_total_limit=8 
#FSDP2
git clone -b benchmark https://github.com/zxd1997066/torchtune.git
git clone https://github.com/pytorch/ao.git
cd ao
pip install -e .
cd ../torchtune
pip install -e .
#meta-llama/Meta-Llama-3.1-8B-Instruct lora
tune run --nproc_per_node 4 lora_finetune_distributed --config llama3_1/8B_lora device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
#meta-llama/Meta-Llama-3.1-8B-Instruct qlora
tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama3_1/8B_qlora_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
#meta-llama/Meta-Llama-3-8B-Instruct dora
tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama3/8B_dora device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=256
#meta-llama/Meta-Llama-3.2-1B-Instruct knowledge_distillation
tune run --nnodes 1 --nproc_per_node 2 knowledge_distillation_distributed --config llama3_2/8B_to_1B_KD_lora_distributed device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
#meta-llama/Meta-Llama-3.1-8B-Instruct lora dpo
tune run --nnodes 1 --nproc_per_node 4 lora_dpo_distributed --config llama3_1/8B_lora_dpo  device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 tokenizer.max_seq_len=256
#meta-llama/Meta-Llama-3.1-8B-Instruct full dpo
tune run --nnodes 1 --nproc_per_node 8 full_dpo_distributed --config llama3_1/8B_full_dpo  device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 tokenizer.max_seq_len=256
#meta-llama/Meta-Llama-3.1-8B-Instruct full finetune
#remove https://github.com/zxd1997066/torchtune/blob/benchmark/recipes/configs/llama3_1/8B_full.yaml#L22-L23
tune run --nproc_per_node 4 full_finetune_distributed --config llama3_1/8B_full device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
#TP
#meta-llama/Meta-Llama-3.1-8B-Instruct full finetune
tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
#DDP
wget https://github.com/zxd1997066/frameworks.ai.pytorch.gpu-models/raw/master/resnet50/main.py
mpiexec -np 8 -ppn 8 python main.py -a resnet50 -b 256 --xpu 0 --dummy --num-iterations 20 -j 12 --bf16 1 --bucket-cap 200 --disable-broadcast-buffers --large-first-bucket --use-gradient-as-bucket-view --seed 123 --dist-backend xccl
#PP
#GPT2ForSequenceClassification
torchrun --nproc-per-node 4 pippy_gpt2.py
#meta-llama/Llama-2-7b-chat-hf
pip uninstall trl
pip install transformers==4.36.2
torchrun --nproc-per-node 2 pippy_llama.py
#Deepspeed
#Deepspeed build need IPEX, but you can uninstall IPEX after installing Deepspeed
#https://github.com/ys950902/DeepSpeed/blob/sy/xccl_enable/accelerator/xpu_accelerator.py#L309-L317 no DpcppBuildExtension in torch.utils.cpp_extension
git clone -b sy/xccl_enable https://github.com/ys950902/DeepSpeed.git
cd DeepSpeed/
pip install py-cpuinfo
pip install -e .
pip install SentencePiece
git clone https://github.com/zxd1997066/frameworks.ai.pytorch.gpu-models.git
cd frameworks.ai.pytorch.gpu-models/LLM/generation
bash run_benchmark_ds.sh
#torchtune single device
tune run full_finetune_single_device --config llama3_1/8B_full_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 optimizer._component_=torchao.prototype.low_bit_optim.AdamWFp8 seed=123 dataset.packed=True tokenizer.max_seq_len=512
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
tune run lora_finetune_single_device --config llama3/8B_dora_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=256
tune run lora_dpo_single_device --config llama3_1/8B_lora_dpo_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 tokenizer.max_seq_len=256
tune run knowledge_distillation_single_device --config llama3_2/8B_to_1B_KD_lora_single_device device=xpu dtype=bf16 max_steps_per_epoch=10 seed=123 dataset.packed=True tokenizer.max_seq_len=512
 
