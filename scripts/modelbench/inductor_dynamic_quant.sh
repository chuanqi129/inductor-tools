#!/bin/bash
LOG_DIR=${1:-inductor_log}
mkdir -p $LOG_DIR
# Prepare transformers && accelerate

rm -rf transformers accelerate

git clone -b test https://github.com/chuanqi129/transformers && cd transformers && \
    python setup.py bdist_wheel && pip install --force-reinstall dist/*.whl && cd ..

git clone -b test https://github.com/chuanqi129/accelerate && cd accelerate && \
    python setup.py bdist_wheel && pip install --no-deps --force-reinstall dist/*.whl && cd ..

# Install requirements for each task
pip install -r transformers/examples/pytorch/text-classification/requirements.txt

bash hf_quant_test.sh

mv logs/ /workspace/pytorch/$LOG_DIR/dynamic_quant/