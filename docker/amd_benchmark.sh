export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

rm .userbenchmark/cpu -rf

log_dir="amd_benchmark"
rm -rf ${log_dir}
mkdir ${log_dir}

echo running cpu userbenchmark........
# precisions="fp32 bf16 amp_bf16 fx_int8"
# models="resnet50,hf_DistilBert,hf_Bert_large,BERT_pytorch,hf_GPT2_large,timm_vision_transformer_large,dcgan"
precisions="fp32"
models="attention_is_all_you_need_pytorch,functorch_maml_omniglot,mnasnet1_0,mobilenet_v3_large,pytorch_unet,shufflenet_v2_x1_0,pyhpc_isoneutral_mixing,LearningToPaint"

for precision in ${precisions}; do
        cmd_prefix="python run_benchmark.py cpu -m ${models} --precision ${precision} --channels-last --launcher --launcher-args=\"--ncores-per-instance=32\""

        # eager
        ${cmd_prefix} 
        mv .userbenchmark/cpu ${log_dir}/${precision}_eager_throughput
        # jit
        ${cmd_prefix}  --jit
        mv .userbenchmark/cpu ${log_dir}/${precision}_jit_throughput
        # jit w/onednn
        ${cmd_prefix}  --jit --fuser fuser3
        mv .userbenchmark/cpu ${log_dir}/${precision}_jit_onednn_throughput
        # inductor
        ${cmd_prefix}  --torchdynamo inductor
        mv .userbenchmark/cpu ${log_dir}/${precision}_inductor_throughput
done
