set +e
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export TORCHINDUCTOR_FREEZING=1

CORES=$(lscpu | grep Core | awk '{print $4}')
end_core=$(expr $CORES - 1)
export OMP_NUM_THREADS=$CORES

LOG_DIR=${1:-gnn_bench}
mkdir -p $LOG_DIR

# uninstall  torch_geometric
pip uninstall torch_geometric -y && pip uninstall ogb -y
# install pyg-lib
# git clone https://github.com/pyg-team/pyg-lib.git && cd pyg-lib
# git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# install torch_geometric
git clone https://github.com/pyg-team/pytorch_geometric && cd pytorch_geometric
git checkout master && git submodule sync && git submodule update --init --recursive && pip install -e . && cd ..
# install ogb
git clone -b yanbing/products_profile https://github.com/yanbing-j/ogb.git && cd ogb && pip install -e . && cd ..
# install pytorch_scatter
git clone https://github.com/rusty1s/pytorch_scatter.git && cd pytorch_scatter
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# install pytorch_sparse
git clone https://github.com/rusty1s/pytorch_sparse.git && cd pytorch_sparse
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# fix issue: AttributeError: module 'importlib.resources' has no attribute 'files'
pip uninstall networkx -y && pip install networkx
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
python -c '''import torch_scatter; print("torch-scatter :", torch_scatter.__version__)''' >>${FILE}
python -c '''import torch_sparse; print("torch-sparse: ", torch_sparse.__version__)''' >>${FILE}

# run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
numactl -C 0-${end_core} --membind=0 python pytorch_geometric/test/nn/models/test_basic_gnn.py --backward --device=cpu 2>&1 | tee -a ${LOG_DIR}/gnn_bench__${timestamp}.log
numactl -C 0-${end_core} --membind=0 python ogb/examples/nodeproppred/products/gnn.py --use_sage --epochs=3 --runs=1 2>&1 | tee -a ${LOG_DIR}/gnn_bench__${timestamp}.log
numactl -C 0-${end_core} --membind=0 python ogb/examples/nodeproppred/products/gnn.py --inference --use_sage --epochs=3 --runs=1 2>&1 | tee -a ${LOG_DIR}/gnn_bench__${timestamp}.log

# get numbers

function parse_data() {
    mode=$1
    row=$2
    column=$3
    res=$(grep "$mode" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*'$mode'://;s/[^0-9.]//' | awk 'NR=='$row'{print}' | awk '{print $'$column'}')
    echo "${res%?}"
}

# case 1
echo GCN_Vanilla_Ftime : $(parse_data "Vanilla" 1 3) >>${FILE}
echo GCN_Vanilla_Btime : $(parse_data "Vanilla" 1 5) >>${FILE}
echo GCN_Vanilla_Ttime : $(parse_data "Vanilla" 1 7) >>${FILE}
echo GCN_Compiled_Ftime : $(parse_data "Compiled" 1 3) >>${FILE}
echo GCN_Compiled_Btime : $(parse_data "Compiled" 1 5) >>${FILE}
echo GCN_Compiled_Ttime : $(parse_data "Compiled" 1 7) >>${FILE}

echo GraphSAGE_Vanilla_Ftime : $(parse_data "Vanilla" 2 3) >>${FILE}
echo GraphSAGE_Vanilla_Btime : $(parse_data "Vanilla" 2 5) >>${FILE}
echo GraphSAGE_Vanilla_Ttime : $(parse_data "Vanilla" 2 7) >>${FILE}
echo GraphSAGE_Compiled_Ftime : $(parse_data "Compiled" 2 3) >>${FILE}
echo GraphSAGE_Compiled_Btime : $(parse_data "Compiled" 2 5) >>${FILE}
echo GraphSAGE_Compiled_Ttime : $(parse_data "Compiled" 2 7) >>${FILE}

echo GIN_Vanilla_Ftime : $(parse_data "Vanilla" 3 3) >>${FILE}
echo GIN_Vanilla_Btime : $(parse_data "Vanilla" 3 5) >>${FILE}
echo GIN_Vanilla_Ttime : $(parse_data "Vanilla" 3 7) >>${FILE}
echo GIN_Compiled_Ftime : $(parse_data "Compiled" 3 3) >>${FILE}
echo GIN_Compiled_Btime : $(parse_data "Compiled" 3 5) >>${FILE}
echo GIN_Compiled_Ttime : $(parse_data "Compiled" 3 7) >>${FILE}

echo EdgeCNN_Vanilla_Ftime : $(parse_data "Vanilla" 4 3) >>${FILE}
echo EdgeCNN_Vanilla_Btime : $(parse_data "Vanilla" 4 5) >>${FILE}
echo EdgeCNN_Vanilla_Ttime : $(parse_data "Vanilla" 4 7) >>${FILE}
echo EdgeCNN_Compiled_Ftime : $(parse_data "Compiled" 4 3) >>${FILE}
echo EdgeCNN_Compiled_Btime : $(parse_data "Compiled" 4 5) >>${FILE}
echo EdgeCNN_Compiled_Ttime : $(parse_data "Compiled" 4 7) >>${FILE}

# speedup

echo GCN_speedup : $(awk 'BEGIN{printf "%.2f\n",'$(parse_data "Vanilla" 1 7)' / '$(parse_data "Compiled" 1 7)'}') >>${FILE}
echo GraphSAGE_speedup : $(awk 'BEGIN{printf "%.2f\n",'$(parse_data "Vanilla" 2 7)' / '$(parse_data "Compiled" 2 7)'}') >>${FILE}
echo GIN_speedup : $(awk 'BEGIN{printf "%.2f\n",'$(parse_data "Vanilla" 3 7)' / '$(parse_data "Compiled" 3 7)'}') >>${FILE}
echo EdgeCNN_speedup : $(awk 'BEGIN{printf "%.2f\n",'$(parse_data "Vanilla" 4 7)' / '$(parse_data "Compiled" 4 7)'}') >>${FILE}

# case 2
# training vanilla
trainning_vanilla_time=$(grep "Time" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Time://;s/[^0-9.]//' | awk 'NR==1{print}')
vanilla_gnn_train_accuracy_use_sage=$(grep "Highest Train:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Train://;s/[^0-9.]//' | awk 'END {print}' | awk 'NF>1{print $1}')
vanilla_gnn_valid_accuracy_use_sage=$(grep "Highest Valid:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Valid://;s/[^0-9.]//' | awk 'END {print}' | awk 'NF>1{print $1}')

echo vanilla_gnn_train_accuracy_use_sage : ${vanilla_gnn_train_accuracy_use_sage} >>${FILE}
echo vanilla_gnn_valid_accuracy_use_sage : ${vanilla_gnn_valid_accuracy_use_sage} >>${FILE}

# training compiled
# detect inductor error
inductor_train_err=$(grep "Error" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Error://;s/[^0-9.]//' | awk 'NR==1{print}')

if [ -n "${inductor_train_err}" ];then
	trainning_compiled_time=0
	compiled_gnn_train_accuracy_use_sage=0
	compiled_gnn_valid_accuracy_use_sage=0
else
	trainning_compiled_time=$(grep "Time" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Time://;s/[^0-9.]//' | awk 'NR==2{print}')
	compiled_gnn_train_accuracy_use_sage=$(grep "Highest Train:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Train://;s/[^0-9.]//' | awk 'NR==3{print}')
	compiled_gnn_valid_accuracy_use_sage=$(grep "Highest Valid:" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Highest Valid://;s/[^0-9.]//' | awk 'NR==3{print}')
fi

echo compiled_gnn_train_accuracy_use_sage : ${compiled_gnn_train_accuracy_use_sage} >>${FILE}
echo compiled_gnn_valid_accuracy_use_sage : ${compiled_gnn_valid_accuracy_use_sage} >>${FILE}

echo trainning_vanilla_time : ${trainning_vanilla_time} >>${FILE}
echo trainning_compiled_time : ${trainning_compiled_time} >>${FILE}


# inference
inductor_infer_err=$(grep "Error" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Error://;s/[^0-9.]//' | awk 'NR==2{print}')

if [ -n "${inductor_infer_err}" ];then
	inference_vanilla_time=$(grep "Time" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Time://;s/[^0-9.]//' | awk 'NR==2{print}')
	inference_compiled_time=0
else
	inference_vanilla_time=$(grep "Time" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Time://;s/[^0-9.]//' | awk 'NR==3{print}')
	inference_compiled_time=$(grep "Time" ${LOG_DIR}/gnn_bench__${timestamp}.log | sed -e 's/.*Time://;s/[^0-9.]//' | awk 'NR==4{print}')
fi

echo inference_vanilla_time : ${inference_vanilla_time} >>${FILE}
echo inference_compiled_time : ${inference_compiled_time} >>${FILE}

echo inductor_compile_err : ${inductor_train_err}${inductor_infer_err} >>${FILE}
