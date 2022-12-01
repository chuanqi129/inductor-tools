#!/usr/bin/bash
set -x

SUITE=${1:-huggingface}
MODEL=${2:-GoogleFnet}
LOG_DIR=${3:-debug}
export TORCHINDUCTOR_TRACE=1

if [[ $USER == "" ]]; then
    USER=root
fi

inductor_codegen_path="/tmp/torchinductor_$USER"
log_path=${LOG_DIR}/${SUITE}_${MODEL}
rm -rf ${log_path}
mkdir -p $log_path
bash ./inductor_single_run.sh ${SUITE} ${MODEL} 2>&1 | tee ${log_path}/raw.log
python ./inductor_cosim.py --dbg_log ${log_path}/raw.log --dbg_code_src_path $inductor_codegen_path --dbg_code_dst_path $log_path/debug --dbg_code_dst_single_readable_py ${log_path}/graph.py
