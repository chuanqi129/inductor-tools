export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
models=alexnet,shufflenet_v2_x1_0,mobilenet_v3_large,vgg16,densenet121,mnasnet1_0,squeezenet1_1,mobilenet_v2,resnet50,resnet152,resnet18,resnext50_32x4d
#models=alexnet,shufflenet_v2_x1_0,mobilenet_v3_large,vgg16,mnasnet1_0,squeezenet1_1,mobilenet_v2,resnet50,resnet152,resnet18,resnext50_32x4d
#models=alexnet
rm -rf .userbenchmark
mkdir inductor_quant/
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+schedule,+inductor,+output_code"
 
TORCHINDUCTOR_FREEZING=1 python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/ptq
rm -rf /tmp/*
TORCHINDUCTOR_FREEZING=1 TORCHINDUCTOR_CPP_WRAPPER=1 python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize --cpp_wrapper --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/cpp
rm -rf /tmp/*
TORCHINDUCTOR_FREEZING=1 python run_benchmark.py cpu -m ${models} --torchdynamo inductor --quantize --is_qat --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/qat
TORCHINDUCTOR_FREEZING=1 python run_benchmark.py cpu -m ${models} --torchdynamo inductor --launcher --launcher-args="--throughput-mode" -b 128 --metrics throughputs
mv .userbenchmark/cpu inductor_quant/general_inductor

mv inductor_quant ../pytorch/$LOG_DIR/
