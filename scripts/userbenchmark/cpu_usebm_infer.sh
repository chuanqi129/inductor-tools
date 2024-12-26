export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
# export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
mkdir userbenchmark_aws/
git clone https://github.com/pytorch/ao.git --depth=1 && cd ao && \
    pip install -e . && cd ..
#inference
echo running cpu userbenchmark........
cmd_prefix='''python run_benchmark.py cpu --test eval --channels-last --metrics throughputs'''
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
#PTQ
export TORCHINDUCTOR_FREEZING=1
numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} ${cmd_prefix} --torchdynamo inductor --quantization pt2e
#numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} ${cmd_prefix} --torchdynamo inductor --quantization auto_quant --inductor-compile-mode max-autotune
#numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} ${cmd_prefix} --torchdynamo inductor --quantization auto_quant
mv .userbenchmark/cpu PT2E
mv PT2E userbenchmark_aws/

numactl -C ${start_core}-${end_core} -m ${mem_allowed_list} ${cmd_prefix} --torchdynamo inductor
mv .userbenchmark/cpu fp32
mv fp32 userbenchmark_aws/
# # FP32 eager
# ${cmd_prefix}
# mv .userbenchmark/cpu eager_throughtput_fp32_infer
# mv eager_throughtput_fp32_infer userbenchmark_aws/

# # BF16 eager
# ${cmd_prefix} --precision amp_bf16
# mv .userbenchmark/cpu eager_throughtput_bf16_infer
# mv eager_throughtput_bf16_infer userbenchmark_aws/

# # fx_int8 eager
# ${cmd_prefix} --precision fx_int8
# mv .userbenchmark/cpu eager_throughtput_fx_int8
# mv eager_throughtput_fx_int8 userbenchmark_aws/

# # FP32 jit with llga:
# ${cmd_prefix} --backend torchscript --fuser fuser3
# mv .userbenchmark/cpu jit_llga_throughtput_fp32
# mv jit_llga_throughtput_fp32 userbenchmark_aws/

# # bf16 jit with llga:
# ${cmd_prefix} --precision amp_bf16 --backend torchscript --fuser fuser3
# mv .userbenchmark/cpu jit_llga_throughtput_amp_bf16
# mv jit_llga_throughtput_amp_bf16 userbenchmark_aws/

# #training
# cmd_prefix='''python run_benchmark.py cpu -m BERT_pytorch,Background_Matting,LearningToPaint,Super_SloMo,alexnet,basic_gnn_edgecnn,basic_gnn_gcn,basic_gnn_gin,basic_gnn_sage,dcgan,demucs,densenet121,dlrm,drq,functorch_dp_cifar10,functorch_maml_omniglot,hf_Albert,hf_Bert,hf_Bert_large,hf_BigBird,hf_DistilBert,hf_GPT2,hf_Longformer,hf_Reformer,lennard_jones,maml_omniglot,mnasnet1_0,mobilenet_v2,mobilenet_v2_quantized_qat,mobilenet_v3_large,nvidia_deeprecommender,phlippe_densenet,phlippe_resnet,pytorch_CycleGAN_and_pix2pix,pytorch_stargan,pytorch_unet,resnet152,resnet18,resnet50,resnet50_quantized_qat,resnext50_32x4d,shufflenet_v2_x1_0,soft_actor_critic,speech_transformer,squeezenet1_1,timm_efficientnet,timm_nfnet,timm_regnet,timm_resnest,timm_vision_transformer,timm_vovnet,tts_angular,vgg16,vision_maskrcnn,yolov3 --test train --channels-last --launcher --launcher-args="--throughput-mode" --metrics throughputs'''
# #OOM still exists on baremetal instance
# #cmd_prefix='''python run_benchmark.py cpu --test train --channels-last --launcher --launcher-args="--throughput-mode" --metrics throughputs'''

# # FP32 eager
# ${cmd_prefix}
# mv .userbenchmark/cpu eager_throughtput_fp32_train
# mv eager_throughtput_fp32_train userbenchmark_aws/

# # BF16 eager
# ${cmd_prefix} --precision amp_bf16
# mv .userbenchmark/cpu eager_throughtput_bf16_train
# mv eager_throughtput_bf16_train userbenchmark_aws/

cp -r userbenchmark_aws ../pytorch/$LOG_DIR/

