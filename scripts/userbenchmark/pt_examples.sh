#!/usr/bin/bash
# set -x
RESULT_DIR=${1:-/tmp/examples_log}
echo "RESULT_DIR: ${RESULT_DIR}"

git clone https://github.com/pytorch/examples.git
EXAMPLES_DIR=$(pwd)/examples

sudo apt-get install bc
sudo apt-get install --reinstall time
TIMEFORMAT=%R

# run mnist
mkdir -p "${RESULT_DIR}/mnist"
cd "${EXAMPLES_DIR}/mnist" || exit
LOG_FILE=${RESULT_DIR}/mnist/result.log
exec_time=$({ time python main.py --epochs 3 >"${LOG_FILE}" 2>&1; } 2>&1)
echo "mnist elapsed time: ${exec_time}s" | tee -a "${LOG_FILE}"

# run mnist-hogwild
mkdir -p ${RESULT_DIR}/mnist_hogwild
cd "${EXAMPLES_DIR}/mnist_hogwild" || exit
LOG_FILE=${RESULT_DIR}/mnist_hogwild/result.log
exec_time=$({ time python main.py --epochs 3 >"${LOG_FILE}" 2>&1; } 2>&1)
echo "mnist-hogwild elapsed time: ${exec_time}s" | tee -a "${LOG_FILE}"

# run CPU WLM LSTM
mkdir -p ${RESULT_DIR}/wlm_cpu_lstm
cd "${EXAMPLES_DIR}/word_language_model" || exit
LOG_FILE=${RESULT_DIR}/wlm_cpu_lstm/result.log
exec_time=$({ time python main.py --epochs 3 --model LSTM >"${LOG_FILE}" 2>&1; } 2>&1)
echo "WLM LSTM elapsed time: ${exec_time}s" | tee -a "${LOG_FILE}"

# run CPU WLM Transformer
mkdir -p ${RESULT_DIR}/wlm_cpu_trans
cd "${EXAMPLES_DIR}/word_language_model" || exit
LOG_FILE=${RESULT_DIR}/wlm_cpu_trans/result.log
exec_time=$({ time python main.py --epochs 3 --model Transformer >"${LOG_FILE}" 2>&1; } 2>&1)
echo "WLM Transformer elapsed time: ${exec_time}s" | tee -a "${LOG_FILE}"
