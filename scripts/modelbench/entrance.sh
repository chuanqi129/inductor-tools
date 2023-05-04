set +e
TAG=${1:-ww18.4}

# kill unused process
itm_1=`ps -ef | grep entrance.sh | awk '{print $2}'`
itm_2=`ps -ef | grep launch.sh | awk '{print $2}'`
itm_3=`ps -ef | grep inductor_test.sh | awk '{print $2}'`

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

# rm finished.txt file
if [ -f finished.txt ]; then
    rm finished.txt
fi

# launch benchmark
bash launch.sh ${TAG}

# create finished.txt when finished
if [ $? -eq 0 ]; then
    echo "benchmark finished!"
    touch finished.txt
fi
