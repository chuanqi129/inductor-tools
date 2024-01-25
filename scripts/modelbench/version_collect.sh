BENCH_COMMIT=${1:-fea73cb}

# collect sw info
curdir=$(pwd)
touch ${curdir}/version.csv
cd /workspace/benchmark
echo torchbench,$(git rev-parse --abbrev-ref HEAD),$(git rev-parse --short HEAD) >>${curdir}/version.csv

cd /workspace/pytorch
torch_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "${torch_branch}" == "nightly" ]];then
    echo torch,${torch_branch},$(git log --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1) >>${curdir}/version.csv
else
    echo torch,${torch_branch},$(git rev-parse --short HEAD) >>${curdir}/version.csv
fi

cd /workspace/vision
echo torchvision,$(git rev-parse --abbrev-ref HEAD),$(git rev-parse --short HEAD) >>${curdir}/version.csv

cd /workspace/text
echo torchtext,$(git rev-parse --abbrev-ref HEAD),$(git rev-parse --short HEAD) >>${curdir}/version.csv

cd /workspace/audio
echo torchaudio,$(git rev-parse --abbrev-ref HEAD),$(git rev-parse --short HEAD) >>${curdir}/version.csv

cd /workspace/data
echo torchdata,$(git rev-parse --abbrev-ref HEAD),$(git rev-parse --short HEAD) >>${curdir}/version.csv

echo dynamo_benchmarks,${torch_branch},$DYNAMO_BENCH >>${curdir}/version.csv