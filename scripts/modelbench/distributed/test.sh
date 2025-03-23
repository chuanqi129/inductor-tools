#FSDP
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
cd frameworks.ai.pytorch.ipex-gpu/examples/gpu/llm/fine-tuning
pip install -r requirements.txt
cd Llama3
git clone https://github.com/huggingface/accelerate.git
cd accelerate
#change self.backend = "xccl" in https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L198
pip install -e .
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1
export model="meta-llama/Meta-Llama-3-8B"
accelerate launch --config_file "fsdp_config.yaml" llama3_ft.py --model_name_or_path ${model} --use_flashattn False --bf16 True --max_seq_length 128 --output_dir="output" --evaluation_strategy="epoch" --learning_rate=1e-3 --gradient_accumulation_steps=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs=1 --save_steps=500 --logging_steps=1 --save_total_limit=8
#FSDP2
git clone -b benchmark https://github.com/zxd1997066/torchtune.git
git clone https://github.com/pytorch/ao.git
cd ao
pip install -e .
cd ../torchtune
pip install -e .
tune run --nproc_per_node 4 lora_finetune_distributed --config llama3_1/8B_lora device=xpu dtype=bf16
#TP
tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full device=xpu dtype=bf16
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
