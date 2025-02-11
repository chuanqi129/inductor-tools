env.NODE_LABEL = 'OOB-MCR'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

env.NODE_PROXY = 'http://proxy.ims.intel.com:911'
if ('NODE_PROXY' in params) {
    echo "NODE_PROXY in params"
    if (params.NODE_PROXY != '') {
        env.NODE_PROXY = params.NODE_PROXY
    }
}
echo "NODE_PROXY: $NODE_PROXY"

env.modelids = 'meta-llama/Llama-2-7b-hf'
if ('modelids' in params) {
    echo "modelids in params"
    if (params.modelids != '') {
        env.modelids = params.modelids
        modelids = modelids.split(",")
    }
}
echo "modelids: $modelids"

env.dtypes = 'bfloat16'
if ('dtypes' in params) {
    echo "dtypes in params"
    if (params.dtypes != '') {
        env.dtypes = params.dtypes
        dtypes = dtypes.split(",")
    }
}
echo "dtypes: $dtypes"

env.torchchat_repo = 'https://github.com/pytorch/torchchat.git'
if ('torchchat_repo' in params) {
    echo "torchchat_repo in params"
    if (params.torchchat_repo != '') {
        env.torchchat_repo = params.torchchat_repo
    }
}
echo "torchchat_repo: $torchchat_repo"

env.torchchat_branch= 'main'
if ('torchchat_branch' in params) {
    echo "torchchat_branch in params"
    if (params.torchchat_branch != '') {
        env.torchchat_branch = params.torchchat_branch
    }
}
echo "torchchat_branch: $torchchat_branch"

env.backend= 'compile'
if ('backend' in params) {
    echo "backend in params"
    if (params.backend != '') {
        env.backend = params.backend
        backend = backend.split(",")
    }
}
echo "backend: $backend"

env.profile= 'false'
if ('profile' in params) {
    echo "profile in params"
    if (params.profile != '') {
        env.profile = params.profile
    }
}
echo "profile: $profile"

env.device= 'cpu'
if ('device' in params) {
    echo "device in params"
    if (params.device != '') {
        env.device = params.device
    }
}
echo "device: $device"

env.extra_args= ''
if ('extra_args' in params) {
    echo "extra_args in params"
    if (params.extra_args != '') {
        env.extra_args = params.extra_args
    }
}
echo "extra_args: $extra_args"

env.input_length = ''
if ('input_length' in params) {
    echo "input_length"
    if (params.input_length != '') {
        input_length = params.input_length
        input_length = input_length.split(",")
    }
}
echo "input_length: $input_length"

env.output_length = ''
if ('output_length' in params) {
    echo "output_length"
    if (params.output_length != '') {
        output_length = params.output_length
        output_length = output_length.split(",")
    }
}
echo "output_length: $output_length"

env.build_image= 'False'
if ('build_image' in params) {
    echo "build_image in params"
    if (params.build_image != '') {
        env.build_image = params.build_image
    }
}
echo "build_image: $build_image"

env.docker_name= 'test'
if ('docker_name' in params) {
    echo "docker_name in params"
    if (params.docker_name != '') {
        env.docker_name = params.docker_name
    }
}
echo "docker_name: $docker_name"

env.upload_log= 'False'
if ('upload_log' in params) {
    echo "upload_log in params"
    if (params.upload_log != '') {
        env.upload_log = params.upload_log
    }
}
echo "upload_log: $upload_log"

env.tensor_parallel= 'False'
if ('tensor_parallel' in params) {
    echo "tensor_parallel in params"
    if (params.tensor_parallel != '') {
        env.tensor_parallel = params.tensor_parallel
    }
}
echo "tensor_parallel: $tensor_parallel"

env.tp_sockets= '1'
if ('tp_sockets' in params) {
    echo "tp_sockets in params"
    if (params.tp_sockets != '') {
        env.tp_sockets = params.tp_sockets
        tp_sockets = tp_sockets.split(",")
    }
}
echo "tp_sockets: $tp_sockets"

env.hardware= 'emr'
if ('hardware' in params) {
    echo "hardware in params"
    if (params.hardware != '') {
        env.hardware = params.hardware
    }
}
echo "hardware: $hardware"

env.test_mode= 'performance'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        env.test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"

env.docker_pull= 'True'
if ('docker_pull' in params) {
    echo "docker_pull in params"
    if (params.docker_pull != '') {
        env.docker_pull = params.docker_pull
    }
}
echo "docker_pull: $docker_pull"

env.torchchat_modeldir= '/localdisk/datasets/huggingface/'
if ('torchchat_modeldir' in params) {
    echo "torchchat_modeldir in params"
    if (params.torchchat_modeldir != '') {
        env.torchchat_modeldir = params.torchchat_modeldir
    }
}
echo "torchchat_modeldir: $torchchat_modeldir"

env.http_proxy='http://proxy.ims.intel.com:911'
env.https_proxy='http://proxy.ims.intel.com:911'
env.BASE_IMAGE= 'gar-registry.caas.intel.com/pytorch/pt_inductor:ubuntu_22.04'
env.LOG_DIR = 'torchchat_log'

def cleanup(){
    try {
        retry(3){
            sh'''
                #!/usr/bin/env bash
                docker_ps=`docker ps -a -q`
                if [ -n "${docker_ps}" ];then
                    docker stop ${docker_ps}
                fi
                docker container prune -f
                docker system prune -f

                docker pull ${BASE_IMAGE}
                docker run -t \
                    -u root \
                    -v ${WORKSPACE}:/root/workspace \
                    --privileged \
                    ${BASE_IMAGE} /bin/bash -c "chmod -R 777 /root/workspace"
            '''
        }
        deleteDir()
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

def pruneOldImage(){
    sh '''
        #!/usr/bin/env bash
        old_image_id=`docker images | grep pt_inductor | awk '{print $3}'`
        old_image=`echo $old_image_id | awk '{print $1}'`
        if [ -n "${old_image}" ]; then
            docker rmi -f $old_image
        fi
        docker system prune -f
    '''
}

if (env.build_image == 'True'){
    node(NODE_LABEL){
        stage("get dockerfile"){
            echo 'get dockerfile......'
            sh '''#!/bin/bash
                set -xe
                # Start docker if docker deamon is not running
                if systemctl is-active --quiet docker; then
                    echo "Docker daemon is running...";
                else
                    echo "Starting docker deamon..." 
                    sudo systemctl start docker || true;
                fi
                # Clean up any existing containers
                docker stop $(docker ps -aq) || true
                docker system prune -af
                # Clean up WORKSPACE
                rm -rf ${WORKSPACE}/* || sudo rm -rf ${WORKSPACE}/* || \
                    docker run -i --rm -v ${WORKSPACE}:${WORKSPACE} ubuntu:22.04 rm -rf ${WORKSPACE}/*
            '''
            deleteDir()
            checkout scm     
        }
        stage("build image"){
            retry(3){
                echo 'Building image......'
                sh '''
                #!/usr/bin/env bash
                docker_img_status=`docker manifest inspect gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device}` || true
                if [ "${device}" = "cpu" ];then
                    dockerfile="Dockerfile.cpu"
                elif [ "${device}" = "xpu" ];then
                    dockerfile=Dockerfile.xpu
                elif [ "${device}" = "a100" ];then
                    dockerfile=Dockerfile
                fi
                if [ -z "${docker_img_status}" ];then
                    cp docker/${dockerfile} ./
                    DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg BASE_IMAGE=${BASE_IMAGE} -t gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device} -f ${dockerfile} --target image .
                else
                    echo "gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device} existed, skip build image"
                fi
                '''
            }
        }
        stage('push image') {
            retry(3){
                echo 'push image......'
                sh '''
                #!/usr/bin/env bash
                docker_img_status=`docker manifest inspect gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device}` || true
                if [ -z "${docker_img_status}" ];then
                    docker push gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device}
                else
                    echo "gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device} existed, skip push image"
                fi
                '''
            }
        }
    }
}
node(NODE_LABEL){
    properties(
        [disableConcurrentBuilds(),]
    )
    cleanup()
    deleteDir()
    checkout scm
    currentBuild.displayName = "#${BUILD_NUMBER}-${NODE_LABEL}-${benchmark_script}-${docker_name}"

    stage("Environment Prepare"){
        withEnv(["docker_name=${docker_name}","device=${device}","build_image=${build_image}","test_mode=${test_mode}","build_image=${build_image}","docker_pull=${docker_pull}"]){
            sh '''
                #!/bin/bash
		        set -x
  
                cd ${WORKSPACE}
                export http_proxy=$NODE_PROXY
                export https_proxy=$NODE_PROXY
                export no_proxy=.intel.com
                
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                
                export log_dir=${WORKSPACE}/logs
                mkdir $log_dir
                image_name=gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device}
                if [ "$docker_pull" = "True" ];then
                    docker pull ${image_name}
                else
                    echo "docker pull is False, use local docker image"
                fi
                if [ "$test_mode" = "performance" ];then
                    if [ "$benchmark_script" = "latency" ];then
                        echo 'modelid,benchmark_script,Parallel,input/output,BS,batch_size,first_token_latency,next_token_latency'| tee -a ${log_dir}/summary.log
                    elif [ "$benchmark_script" = "throughput" ];then
                        echo 'modelid,benchmark_script,datasets,Parallel,input/output,num_prompts,token_throughput,request_throughput,first_token_latency,next_token_latency, bs_group' | tee -a ${log_dir}/summary.log
                    elif [ "$benchmark_script" =~ "serving" ];then 
                        echo 'modelid,benchmark_script,datasets,Parallel,input/output,num_prompts,request_rate,token_throughput,first_token_latency,inter_token_latency' | tee -a ${log_dir}/summary.log
                    fi
                else
                    echo 'modelid,benchmark_script,dtype,Parallel,input/output,Status' | tee -a ${log_dir}/summary.log
                fi
            '''
        }//with env

    }//stage env prepare
    
    stage("Benchmark"){
        for (modelid in modelids){
            for (dtype in dtypes){
                for (in_len in input_length){
                    for (ou_len in output_length){
                        for (bak in backend){
                            withEnv(["modelid=${modelid}","dtype=${dtype}","in_len=${in_len}","ou_len=${ou_len}","bak=${bak}","device=${device}"]){
                                sh '''
                                    #!/bin/bash
                                    set -x
                                    if [ "${bak}" = "eager" ];then
                                        prefill="noprefill"
                                        autotune="noautotune"
                                    else
                                        prefill="prefill"
                                        autotune="max_autotune"
                                    fi
                                    docker run -tid --name torchchat_test \
                                        --privileged \
                                        --env https_proxy=${https_proxy} \
                                        --env http_proxy=${http_proxy} \
                                        --env HF_HUB_TOKEN=$HF_TOKEN \
                                        --net host --shm-size 10G \
                                        -v ~/.cache:/root/.cache \
                                        -v ${WORKSPACE}/${LOG_DIR}:/workspace/torchchat/${LOG_DIR} \
                                        -v ${torchchat_modeldir}:/localdisk/datasets/huggingface/
                                        gar-registry.caas.intel.com/pytorch/torchchat:${docker_name}_${device}
                                    docker cp scripts/modelbench/torchchat_cpu.sh torchchat_test:/workspace/torchchat
                                    docker exec -i torchchat_test bash -c "bash torchchat_cpu.sh $dtype $prefill $bak $autotune $profile $modelid $ou_len $in_len "
                                '''
                            }
                        }
                    }
                }
            }
        }

    if (fileExists("${WORKSPACE}/logs/summary.log") == true){
        archiveArtifacts "logs/summary.log"
    }   
    archiveArtifacts artifacts: "**/torchchat_log/**", excludes: null
    fingerprint: true 
    if (env.upload_log == 'True'){
        sh '''
        set -e 
        set -x
        python vllm_func_json.py \
            --file_name ${WORKSPACE}/logs/summary.log --hardware ${hardware}
        '''
    }
    }   
}
