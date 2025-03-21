#FSDP
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
cd frameworks.ai.pytorch.ipex-gpu/examples/gpu/llm/fine-tuning
pip install -r requirements.txt
cd Llama3
git clone https://github.com/huggingface/accelerate.git
cd accelerate
#change self.backend = "xccl" in https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L198
pip install -e .
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