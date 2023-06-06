export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export LRU_CACHE_CAPACITY=1024

rm .userbenchmark/cpu -rf

log_dir="arm_benchmark"
rm -rf ${log_dir}
mkdir ${log_dir}

echo running cpu userbenchmark........
precisions="fp32 bf16 fx_int8"
for precision in ${precisions}; do
	if [ $precision = "fx_int8" ]; then
		precison_ext="fx_int8 --quant-engine qnnpack"
		unset DNNL_DEFAULT_FPMATH_MODE
	elif [ $precision = "bf16" ]; then
		precision_ext="fp32"
		export DNNL_DEFAULT_FPMATH_MODE=BF16
	else
		precision_ext=${precision}
		unset DNNL_DEFAULT_FPMATH_MODE
	fi
        cmd_prefix="python run_benchmark.py cpu -m resnet50,hf_DistilBert,hf_Bert_large,BERT_pytorch,hf_GPT2_large,timm_vision_transformer_large,dcgan --precision ${precision_ext} --channels-last --launcher --launcher-args=\"--throughput-mode\""

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

