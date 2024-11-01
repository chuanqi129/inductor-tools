export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
LOG_DIR=${1:-inductor_log}
cd ../benchmark
mkdir -p $LOG_DIR
mkdir inductor_quant_acc/
pip install networkx==2.8
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+schedule,+inductor,+output_code"

python inductor_quant_acc.py 2>&1 |& tee "./inductor_quant_acc_ptq.log"
mv ./inductor_quant_acc_ptq.log inductor_quant_acc/
rm -rf /tmp/*
TORCHINDUCTOR_CPP_WRAPPER=1 python inductor_quant_acc.py --cpp_wrapper 2>&1 |& tee "./inductor_quant_acc_ptq_cpp_wrapper.log"
mv ./inductor_quant_acc_ptq_cpp_wrapper.log inductor_quant_acc/
rm -rf /tmp/*
python inductor_quant_acc.py  --is_qat 2>&1 |& tee "./inductor_quant_acc_qat.log"
mv ./inductor_quant_acc_qat.log inductor_quant_acc/
python inductor_quant_acc.py --is_fp32 2>&1 |& tee "./inductor_quant_acc_fp32.log"
mv ./inductor_quant_acc_fp32.log inductor_quant_acc/

mv inductor_quant_acc ../pytorch/$LOG_DIR/
