#!/usr/bin/bash
set -x

THREAD=${1:-multiple} # multiple / single / all
MODE=${2:-inference} # inference / training
SCENARIO=${3:-performance} # accuracy / performance
SUITE=${4:-huggingface} # torchbench / huggingface / timm_models
MODEL=${5:-GoogleFnet}
DT=${6:-float32} # float32 / amp
CHANNELS=${7:-first} # first / last
SHAPE=${8:-static} # static / dynamic
WRAPPER=${9:-default} # default / cpp
BS=${10:-0}
LOG_DIR=${11:-debug}
export TORCH_COMPILE_DEBUG=1

if [[ $USER == "" ]]; then
    USER=root
fi

inductor_codegen_path="/tmp/torchinductor_$USER"
log_path=${LOG_DIR}/${SUITE}_${MODEL}
rm -rf ${log_path}
mkdir -p $log_path
bash ./inductor_single_run.sh ${THREAD} ${MODE} ${SCENARIO} ${SUITE} ${MODEL} ${DT} ${CHANNELS} ${SHAPE} ${WRAPPER} ${BS} 2>&1 | tee ${log_path}/raw.log
python ./inductor_cosim.py --dbg_log ${log_path}/raw.log --dbg_code_src_path $inductor_codegen_path --dbg_code_dst_path $log_path/debug --dbg_code_dst_single_readable_py ${log_path}/graph.py
