#!/bin/bash
LOG_DIR=${1:-inductor_log}
mkdir -p $LOG_DIR
mkdir hf_quant
# Prepare transformers && accelerate

rm -rf transformers accelerate

git clone -b test https://github.com/chuanqi129/transformers && cd transformers && \
    python setup.py bdist_wheel && pip install --force-reinstall dist/*.whl && cd ..

git clone -b test https://github.com/chuanqi129/accelerate && cd accelerate && \
    python setup.py bdist_wheel && pip install --no-deps --force-reinstall dist/*.whl && cd ..

# Install requirements for each task
pip install -r transformers/examples/pytorch/text-classification/requirements.txt

bash hf_quant_test.sh text torch_compile
mv logs/ hf_quant/fp32_compile

bash hf_quant_test.sh text torch_compile_quant
mv logs/ hf_quant/dynamic_quant

bash hf_quant_test.sh text torch_compile_quant_static
mv logs/ hf_quant/static_quant

mv hf_quant/ /workspace/pytorch/$LOG_DIR/
