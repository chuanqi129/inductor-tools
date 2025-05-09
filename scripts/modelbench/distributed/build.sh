/opt/python/cp310-cp310/bin/python -m venv /opt/xpu-build
source /opt/xpu-build/bin/activate
pip install -U pip wheel setuptools
export USE_XCCL=1
export USE_ONEMKL=1
cd /var/lib/jenkins/workspace/pytorch/
pip install -r requirements.txt
pip install mkl-static mkl-include
export USE_KINETO=OFF
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/umf/latest/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
source /opt/intel/oneapi/ccl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh
WERROR=1 python setup.py bdist_wheel 2>&1 | tee /var/lib/jenkins/workspace/distributed_log/pytorch_build.log >/dev/null
python -m pip install patchelf
rm -rf ./tmp
bash third_party/torch-xpu-ops/.github/scripts/rpath.sh /var/lib/jenkins/workspace/pytorch/dist/torch*.whl
python -m pip install --force-reinstall tmp/torch*.whl
cd ..
python pytorch/torch/utils/collect_env.py
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"

cp pytorch/tmp/torch*.whl distributed_log/ 
