#!/usr/bin/bash
set -x

SUITE=${1:-huggingface} # torchbench / huggingface / timm_models
MODEL=${2:-GoogleFnet}
DT=${3:-float32} # float32 / amp
CHANNELS=${4:-first} # first / last
SHAPE=${5:-static} # static / dynamic
WRAPPER=${6:-default} # default / cpp
BS=${7:-0}
LOG_DIR=${8:-debug}
export TORCH_COMPILE_DEBUG=1

if [[ $USER == "" ]]; then
    USER=root
fi

inductor_codegen_path="/tmp/torchinductor_$USER"
log_path=${LOG_DIR}/${SUITE}_${MODEL}
rm -rf ${log_path}
mkdir -p $log_path
bash ./inductor_single_run.sh multiple inference performance ${SUITE} ${MODEL} ${DT} ${CHANNELS} ${SHAPE} ${BS} ${WRAPPER} 2>&1 | tee ${log_path}/raw.log
python ./inductor_cosim.py --dbg_log ${log_path}/raw.log --dbg_code_src_path $inductor_codegen_path --dbg_code_dst_path $log_path/debug --dbg_code_dst_single_readable_py ${log_path}/graph.py
