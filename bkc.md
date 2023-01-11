### BKC of Torchinductor Dashboard Regular Refresh 

##### 1. build image / loggin container /run benchmark
Test based on docker image.Releated files:
[Dockerfile](https://github.com/chuanqi129/inductor-tools/blob/main/docker/Dockerfile)
[Benchmark script](https://github.com/chuanqi129/inductor-tools/blob/main/scripts/modelbench/inductor_test.sh)

steps:
```
cd ~/docker

## docker image build
DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg PT_REPO=https://github.com/pytorch/pytorch.git --build-arg PT_BRANCH=nightly --build-arg PT_COMMIT=fac4361 --build-arg https_proxy=${https_proxy} -t pt_inductor:ww02.3 -f Dockerfile --target image .

## docker container create
docker run -tid --name inductor_ww02.2 --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/ubuntu/docker/inductor_log:/workspace/pytorch/inductor_log pt_inductor:ww02.2

## loggin container
docker exec -ti inductor_ww02.2 bash

## run benchmark
bash inductor_test.sh
```
##### 2. publish results / generate report

Find and download result from ```/home/ubuntu/docker/inductor_log```

Expected directory structure:

```
WW02.2/inductor_log/
|---multi_threads/single_thread_cf_logs_*/
      |---passrate.csv
      |---geomean.csv
      |---gh_executive_summary.txt
      |---gh_inference.txt
      |---gh_title.txt
      |---inductor_{suite}_float32_inference_cpu_accuracy.csv
      |---inductor_{suite}_float32_inference_cpu_performance.csv
|---multi_threads/single_thread_model_bench_log_*.log
```

##### a) For publish into dashboard:

Please copy **gh_title.txt**  **gh_executive_summary.txt** and **gh_inference.txt** into one file successively from multi_threads_cf_logs_*/ folder for multi threads and add workday in title and updtae SW info in this two files.
Single thread does similar thing.


##### b)For report generate:
You can use [log_parser.py](https://github.com/chuanqi129/inductor-tools/blob/b2c1284c7d33144db8ca619c1e23c0cea954c5d4/scripts/modelbench/log_parser.py) for report generation

```
python log_parser.py --reference WW48.2 --target WW48.4
python log_parser.py --target WW48.4
```

Please confirm torchbench model passrate due to some models Not implemented in CPU but count into calculation.(like tacotron2)