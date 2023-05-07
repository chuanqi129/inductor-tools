set +e
TAG=${1:-ww18.4}
PRECISION=${2:-float32}

TORCH_REPO=${3:-https://github.com/pytorch/pytorch.git}
TORCH_BRANCH=${4:-nightly}
TORCH_COMMIT=${5:nightly}
DYNAMO_BENCH=${6:-fea73cb}

# kill unused process
itm_1=`ps -ef | grep entrance_train.sh | awk '{print $2}'`
itm_2=`ps -ef | grep launch_train.sh | awk '{print $2}'`
itm_3=`ps -ef | grep inductor_train.sh | awk '{print $2}'`

if [ -n "${itm_1}" ]; then
    sudo kill -9 $item_1
fi

if [ -n "${itm_2}" ]; then
    sudo kill -9 $item_2
fi

if [ -n "${itm_3}" ]; then
    sudo kill -9 $item_3
fi

# cd target dir
echo cur_dir :`pwd`
cd /home/ubuntu/docker

# rm finished_train.txt file
if [ -f finished_train.txt ]; then
    rm finished_train.txt
fi

# launch benchmark
bash launch_train.sh ${TAG} ${PRECISION} ${TORCH_REPO} ${TORCH_BRANCH} ${TORCH_COMMIT} ${DYNAMO_BENCH}

# create finished.txt when finished
if [ $? -eq 0 ]; then
    echo "training benchmark finished!"
    touch finished_train.txt
fi
