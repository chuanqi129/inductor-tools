export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
mkdir userbenchmark_aws/

echo running cpu userbenchmark........
cmd_prefix='''python run_benchmark.py cpu -m BERT_pytorch,Background_Matting,LearningToPaint,Super_SloMo,alexnet,basic_gnn_edgecnn,basic_gnn_gcn,basic_gnn_gin,basic_gnn_sage,dcgan,demucs,densenet121,dlrm,drq,functorch_dp_cifar10,functorch_maml_omniglot,hf_Albert,hf_Bert,hf_Bert_large,hf_BigBird,hf_DistilBert,hf_GPT2,hf_Longformer,hf_Reformer,lennard_jones,maml_omniglot,mnasnet1_0,mobilenet_v2,mobilenet_v2_quantized_qat,mobilenet_v3_large,nvidia_deeprecommender,phlippe_densenet,phlippe_resnet,pytorch_CycleGAN_and_pix2pix,pytorch_stargan,pytorch_unet,resnet152,resnet18,resnet50,resnet50_quantized_qat,resnext50_32x4d,shufflenet_v2_x1_0,soft_actor_critic,speech_transformer,squeezenet1_1,timm_efficientnet,timm_nfnet,timm_regnet,timm_resnest,timm_vision_transformer,timm_vovnet,tts_angular,vgg16,vision_maskrcnn,yolov3 --test train --channels-last --launcher --launcher-args="--throughput-mode" --metrics throughputs'''

# FP32 eager
${cmd_prefix}
mv .userbenchmark/cpu eager_throughtput_fp32_train
mv eager_throughtput_fp32_train userbenchmark_aws/

# BF16 eager
${cmd_prefix} --precision amp_bf16
mv .userbenchmark/cpu eager_throughtput_bf16_train
mv eager_throughtput_bf16_train userbenchmark_aws/

mv userbenchmark_aws ../pytorch/$LOG_DIR/

