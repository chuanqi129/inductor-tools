### BKC of Torchinductor Dashboard Regular Refresh 

<details>
<summary>1. loggin in benchmark machine</summary>

please use mlt-ace machine as jump server to access aws instance.

steps:

##### a) new a config file under .ssh/ dir,content as follows(please specify lasted Hostname and your actual IdentityFile):
(-rw-r--r-- 1 .ssh/config)

```
Host icx-2
Hostname ec2-3-234-221-190.compute-1.amazonaws.com
User ubuntu
IdentityFile /home2/bzheng/icx-key.pem
ProxyCommand nc --proxy-type socks4 --proxy proxy-shz.intel.com:1080 %h %p
Port 22
```
##### b). new icx-key.pem file,content as follows:
    (-r-------- 1 icx-key.pem)

```

-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAoNA8CfVj9R5V05fVO8z25jYHoXs8lQN2Eqzhj3wZjXxVylr8
PMdd6H+rxa6dK+VVz3bdB7DdimM3hY1341sDZ7t0hvzzFfhvIG0+MuZG/mj49/zl
S+Thy/GJVo2sd308Z0MF/OAeiT54gCelZjd+uI9RRoytMpDbIxENLSbn/ViIBfrn
Dz1zCsTTYENIBJmLzn9Za9LjjojgoqWO5vEDxfgxd4TOG9GirBHUrtSC9QlIzNg9
WkRsjj41Aqr7KKKWu5Ifd/Rkg5IgKiAL22828NdAAPtYDclJBZ9hIiQaagVt3CPS
YX8e8/WypQRcmrbBUA7jSe1ycZMhqxgza1YpPQIDAQABAoIBAF6B4jbsCuF8AW6H
lZ4+V391k0h4d2MPlK1nAmxjv1SiFH0GuaQyemLv3RAhDWdFsQdq0Hz7mUcCBhgr
yqBH1Zl51gKa13U6+nvxF5OsN16+bEq5Sjwu9+/2NAM04X2bkQsSnYs+X1PC7ehK
5G5NAOEQAPDYLdVAwjg6hibZ7Y66L+0owj9egpQ9B9nTy7p9Wq7f8+7y3E38PMYI
am/5Vwea3jREUYf0noxXj/WTFJ+EdvDUQMMDp2MDiBYAwIKDxKPrTPhKWQ1zlHMV
H8QCYF0BrxxPcU8D0arhrD4A0l/zHf0q7deAppkWrrbbtMiFl7f8dd5+iOYhSQs3
NZpRWQECgYEA4YLUdn3H0jwKVn6tnCn0GIp64y8TrqFIYTOVJ+AmsItgtqhviHdA
CN0z1HUrF+tG+xqjMVZBlfGIaZwhgRRS8t34UNb5RY5FOyWdWG1tzYFJ0nDp1V2b
2U94S6oXXWl8PoaUlrnJajPAiwu0acXVpFKZ2PPBh1QJQCbPYChEb0ECgYEAto4n
EzoVMM38k3KowdHaIZ87vNOYtP9hde8POr/2w+zRktzWomIevc6j9Zid/bOaZK9w
3SXHz/sbpGaC8kOpKFuLRXt26iGOlBvdC3BHG3ktONPQrjb8ens0+YqZtq0Hg35d
UNeVSFE5jKwt/h3CMpMsnkRookuy+giG2O2ctv0CgYEAvqWpUsdBYDXME5Wi1gwN
xZWFEH1jKVZGZ46BQaXZb5VVXPaG3BrcsuG3CJ1Joj4Q0waRAovGhKoeRhJtrL9u
5M3NOSKUuD3vv3IsVG6mzn7H1zYIKY6qzAvISkdw21Lra7zA76//XF6GH8b0bhuH
YxVF5Uklym+8JlTpKoTAXkECgYEAsjAEwgaionWdQMrdH/pWoRTM+V3MI3dWyQdj
5SN0mP7x/Rjjw7JkcT5e2Z1zMSx62iZI0wNKip5+97Q7wn76gPMB+PsvMg85HSYp
Vz3rmkxoMJzHhxUS0faq2ZxIt38i412Xdg5Hn0PxGXcCmZSwdi8jkQQS4b9j3hwu
i1U6730CgYABUt2A1AVA0+acI9g+wMdmAqVro5M5k9npc3JBc72yoQiVpl2xO/eb
Xs+6cdmnGolLaLrRTY+/GMCuR1ikory/Vq9NZS4zZQ1UMsBFwfLESCiAFY1mDgWs
GEhocBEjbigzCwis8zqTbU4FpHtV6IJuv79ibjRNTN3ACGQ1ItCukw==
-----END RSA PRIVATE KEY-----

```
##### c). ssh ubuntu@icx-2

##### d). tmux a -t 0

</details>


<details>
<summary>2. build image / loggin container /run benchmark</summary>
Working directory is /home/ubuntu/docker

For image build:  /home/ubuntu/docker/Dockerfile
Benchmark script: /home/ubuntu/docker/inductor_test.sh
Benchmark result: /home/ubuntu/docker/inductor_log

Steps:

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
</details>

<details>
<summary>3. Publish results / Generate report</summary>

Find and download result from /home/ubuntu/docker/inductor_log

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

##### For publish into dashboard:

Please copy **gh_title.txt**  **gh_executive_summary.txt** and **gh_inference.txt** into one file successively from multi_threads_cf_logs_*/ folder for multi threads.
Single thread does similar thing.

Note: 
please add workday in title and updtae SW info in this two files.

##### For report generate:
You can use [log_parser.py](https://github.com/chuanqi129/inductor-tools/blob/b2c1284c7d33144db8ca619c1e23c0cea954c5d4/scripts/modelbench/log_parser.py) for report generation

```
python log_parser.py --reference WW48.2 --target WW48.4
python log_parser.py --target WW48.4
```

Note: 
Please confirm torchbench model passrate due to some models Not implemented in CPU but count into calculation.(like tacotron2)

</details>