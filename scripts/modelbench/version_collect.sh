BENCH_COMMIT=${1:-fea73cb}

# collect sw info
curdir=$(pwd)
mkdir -p ${curdir}/${LOG_DIR}
touch ${curdir}/${LOG_DIR}/version.csv
# Title
echo name,branch,commit >>${curdir}/${LOG_DIR}/version.csv
cd /workspace/benchmark
echo torchbench,main,$(git rev-parse --short HEAD) >>${curdir}/${LOG_DIR}/version.csv

cd /workspace/pytorch
torch_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "${torch_branch}" == "nightly" ]] || [[ "${torch_branch}" == "HEAD" ]];then
    echo torch,main,$(git log --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1) >>${curdir}/${LOG_DIR}/version.csv
else
    echo torch,${torch_branch},$(git rev-parse --short HEAD) >>${curdir}/${LOG_DIR}/version.csv
fi

echo torchvision,main,$(python -c "import torchvision; print(torchvision.__version__)") >>${curdir}/${LOG_DIR}/version.csv
echo torchtext,main,$(python -c "import torchtext; print(torchtext.__version__)") >>${curdir}/${LOG_DIR}/version.csv
echo torchaudio,main,$(python -c "import torchaudio; print(torchaudio.__version__)") >>${curdir}/${LOG_DIR}/version.csv
echo torchdata,main,$(python -c "import torchdata; print(torchdata.__version__)") >>${curdir}/${LOG_DIR}/version.csv
echo dynamo_benchmarks,main,$DYNAMO_BENCH >>${curdir}/${LOG_DIR}/version.csv
