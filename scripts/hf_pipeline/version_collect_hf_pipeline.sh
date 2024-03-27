LOG_DIR=${1:-hf_pipeline_log}

# collect sw info
curdir=$(pwd)
mkdir -p ${curdir}/${LOG_DIR}
touch ${curdir}/${LOG_DIR}/version.csv
# Title
echo name,branch,commit >> ${curdir}/${LOG_DIR}/version.csv
cd /workspace/pytorch
torch_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "${torch_branch}" == "nightly" ]] || [[ "${torch_branch}" == "HEAD" ]];then
    echo torch,main,$(git log --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1) >>${curdir}/${LOG_DIR}/version.csv
else
    echo torch,${torch_branch},$(git rev-parse --short HEAD) >>${curdir}/${LOG_DIR}/version.csv
fi

echo transformers,main,$(python -c "import transformers; print(transformers.__version__)") >>${curdir}/${LOG_DIR}/version.csv