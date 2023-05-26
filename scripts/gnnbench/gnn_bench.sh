export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

LOG_DIR=${1:-gnn_bench}
mkdir -p $LOG_DIR

# uninstall pyg-lib torch_geometric
pip uninstall pyg-lib -y && pip uninstall torch_geometric -y && pip uninstall ogb -y
# install pyg-lib from source
git clone https://github.com/pyg-team/pyg-lib.git && cd pyg-lib
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# install torch_geometric from source
git clone https://github.com/pyg-team/pytorch_geometric && cd pytorch_geometric
git checkout master && git submodule sync && git submodule update --init --recursive && pip install -e . && cd ..
# install ogb
git clone https://github.com/snap-stanford/ogb && cd ogb && pip install -e . && cd ..

# collect sw info
curdir=$(pwd)
FILE=${curdir}/${LOG_DIR}/result.txt
if [ -f ${FILE} ]; then
    rm ${FILE}
fi
touch ${FILE}

cd /workspace/benchmark
echo torchbench : $(git rev-parse --short HEAD) >>${FILE}
cd /workspace
python -c '''import torch,torchvision,torchtext,torchaudio,torchdata,torch_geometric,pyg_lib; \
        print("torch : ", torch.__version__); \
        print("torchvision : ", torchvision.__version__); \
        print("torchtext : ", torchtext.__version__); \
        print("torchaudio : ", torchaudio.__version__); \
        print("torchdata : ", torchdata.__version__); \
        print("pyg_lib : ", pyg_lib.__version__); \
        print("torch_geometric : ", torch_geometric.__version__);''' >>${FILE}
python -c '''import ogb; print("ogb : ", ogb.__version__)''' >>${FILE}

# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
numactl -C 0-${end_core} --membind=0 python pytorch_geometric/test/nn/models/test_basic_gnn.py --backward --device=cpu 2>&1 | tee -a ${LOG_DIR}/gnn_bench__${timestamp}.log
numactl -C 0-${end_core} --membind=0 python ogb/examples/nodeproppred/products/gnn.py --use_sage --epochs=3 --runs=1 2>&1 | tee -a ${LOG_DIR}/gnn_bench__${timestamp}.log

# get numbers

# case 1
GCN_Vanilla_time=$(grep "Vanilla" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Vanilla://;s/[^0-9.]//' | awk 'NR==1{print}' | awk '{print $7}')
GraphSAGE_Vanilla_time=$(grep "Vanilla" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Vanilla://;s/[^0-9.]//' | awk 'NR==2{print}' | awk '{print $7}')
GIN_Vanilla_time=$(grep "Vanilla" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Vanilla://;s/[^0-9.]//' | awk 'NR==3{print}' | awk '{print $7}')
EdgeCNN_Vanilla_time=$(grep "Vanilla" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Vanilla://;s/[^0-9.]//' | awk 'NR==4{print}' | awk '{print $7}')

GCN_Compiled_time=$(grep "Compiled" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Compiled://;s/[^0-9.]//' | awk 'NR==1{print}' | awk '{print $7}')
GraphSAGE_Compiled_time=$(grep "Compiled" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Compiled://;s/[^0-9.]//' | awk 'NR==2{print}' | awk '{print $7}')
GIN_Compiled_time=$(grep "Compiled" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Compiled://;s/[^0-9.]//' | awk 'NR==3{print}' | awk '{print $7}')
EdgeCNN_Compiled_time=$(grep "Compiled" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Compiled://;s/[^0-9.]//' | awk 'NR==4{print}' | awk '{print $7}')

# case 2
gnn_train_accuracy_use_sage=$(grep "Highest Train:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Train://;s/[^0-9.]//' | awk 'NR==1{print}')
gnn_valid_accuracy_use_sage=$(grep "Highest Valid:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Valid://;s/[^0-9.]//' | awk 'NR==1{print}')

echo GCN_Vanilla_time : ${GCN_Vanilla_time} >>${FILE}
echo GCN_Compiled_time : ${GCN_Compiled_time} >>${FILE}

echo GraphSAGE_Vanilla_time : ${GraphSAGE_Vanilla_time} >>${FILE}
echo GraphSAGE_Compiled_time : ${GraphSAGE_Compiled_time} >>${FILE}

echo GIN_Vanilla_time : ${GIN_Vanilla_time} >>${FILE}
echo GIN_Compiled_time : ${GIN_Compiled_time} >>${FILE}

echo EdgeCNN_Vanilla_time : ${EdgeCNN_Vanilla_time} >>${FILE}
echo EdgeCNN_Compiled_time : ${EdgeCNN_Compiled_time} >>${FILE}

echo gnn_train_accuracy_use_sage : ${gnn_train_accuracy_use_sage} >>${FILE}
echo gnn_valid_accuracy_use_sage : ${gnn_valid_accuracy_use_sage} >>${FILE}
