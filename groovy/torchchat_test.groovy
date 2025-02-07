NODE_LABEL = 'OOB-MCR'
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

env.prefill = 'noprefill'
if ('prefill' in params) {
    echo "prefill in params"
    if (params.prefill != '') {
        env.prefill = params.prefill
    }
}
echo "prefill: $prefill"

env.conda_name = 'tgi'
if ('conda_name' in params) {
    echo "conda_name in params"
    if (params.conda_name != '') {
        env.conda_name = params.conda_name
    }
}
echo "conda_name: $conda_name"

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
    }
}
echo "backend: $backend"

env.autotune= 'max_autotune'
if ('autotune' in params) {
    echo "autotune in params"
    if (params.autotune != '') {
        env.autotune = params.autotune
    }
}
echo "autotune: $autotune"

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

input_length = ''
if ('input_length' in params) {
    echo "input_length"
    if (params.input_length != '') {
        input_length = params.input_length
        input_length = input_length.split(",")
    }
}
echo "input_length: $input_length"

output_length = ''
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

env.length_configs= ' '
if ('length_configs' in params) {
    echo "length_configs in params"
    if (params.length_configs != '') {
        env.length_configs = params.length_configs
        length_configs = length_configs.split(",")
    }
}
echo "length_configs: $length_configs"

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

env.http_proxy='http://proxy.ims.intel.com:911'
env.https_proxy='http://proxy.ims.intel.com:911'
env.BASE_IMAGE= 'gar-registry.caas.intel.com/pytorch/pt_inductor:ubuntu_22.04'

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
    node(IMAGE_NODE){
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
                    withEnv(["modelid=${modelid}","dtype=${dtype}","device=${device}"]){
                        sh '''
                            #!/bin/bash

                            set -x
                            if [ "$device" = "cpu" ];then


                                vllm_server_script="vllm_server_launch.sh"
                            else
                                vllm_server_script="vllm_server_launch.sh"
                            fi
                            if [ "$benchmark_script" = "serving" ] || [ "$benchmark_script" = "llmperf_serving" ];then
                                container_name=vllm-$dtype
                                docker stop $(docker ps -a -q) || true
                                docker rm $(docker ps -a -q) || true

                                if [ "${device}" = "xpu" ];then
                                    docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
                                        -e http_proxy=${http_proxy:-"http://proxy-dmz.intel.com:912"} \
                                        -e https_proxy=${http_proxy:-"http://proxy-dmz.intel.com:912"} \
                                        -v ${hugginface_path}:/root/.cache/ \
                                        -v ${WORKSPACE}:/workspace \
                                        -v /dev/dri/by-path:/dev/dri/by-path \
                                        --device /dev/dri:/dev/dri \
                                        --name=${container_name} \
                                        --entrypoint='' \
                                        gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name}  \
                                        /bin/bash -c "bash /workspace/${vllm_server_script} $modelid $dtype $device $tp_socket ${hardware} ${extra_args}"
                                else
                                    docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
                                        -e http_proxy=${http_proxy:-"http://proxy-dmz.intel.com:912"} \
                                        -e https_proxy=${http_proxy:-"http://proxy-dmz.intel.com:912"} \
                                        -v ${hugginface_path}:/root/.cache/ \
                                        -v ${WORKSPACE}:/workspace \
                                        --name=${container_name} \
                                        --entrypoint='' \
                                        gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name}  \
                                        /bin/bash -c "bash /workspace/${vllm_server_script} $modelid $dtype $device $tp_socket ${engine_type} ${extra_args}"
                                fi
                                # docker exec -t -d "${container_name}" /bin/bash /workspace/${vllm_server_script} $modelid $dtype $device $tp_socket ${hardware} ${extra_args}

                                sleep 30s
                                connected=0
				                connect_info="Application startup complete"
                              
                                for i in 1 3 5 10 20
                                do
                                    if docker logs ${container_name} 2>&1 | grep -q "${connect_info}"; then
                                        connected=1
                                        echo "vLLM service has been LAUNCHED!. Proceeding to the next step."
                                        break
                                    elif docker logs ${container_name} 2>&1 | grep -q "RPCServer process died before responding to readiness probe"; then
                                        echo "vLLM service failed to launch"
                                        break
                                    else
                                        sleep_time=$(( $i * 60 ))
                                        echo "vLLM service has not yet been launched! Checking again in $sleep_time seconds..."
                                        sleep $sleep_time
                                    fi
                                done
                                docker logs ${container_name} | tee -a ${WORKSPACE}/logs/${tp_socket}_${dtype}_server.log
                            fi
                        '''
                    } // with env
                    for (length_config in length_configs){
                        for (num_prompt in num_prompts){
                            for (request_rate in request_rates){
                                withEnv(["modelid=${modelid}","num_prompt=${num_prompt}","dtype=${dtype}","backend=${backend}","device=${device}","dataset=${dataset}","test_mode=${test_mode}",\
                                "docker_name=${docker_name}","conda_name=${conda_name}","benchmark_script=${benchmark_script}","request_rate=${request_rate}","OPENAI_API_BASE=${OPENAI_API_BASE}",\
                                "tp_socket=${tp_socket}","tensor_parallel=${tensor_parallel}","extra_args=${extra_args}","length_config=${length_config}","hardware=${hardware}","tp_backbone=${tp_backbone}"]){
                                    docker.image("gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name}").inside(" \
                                        --shm-size 10g \
                                        --net=host \
                                        -v ${WORKSPACE}:/workspace \
                                        -u root \
                                        --privileged \
                                        --entrypoint='' \
                                        -v ${hugginface_path}:/root/.cache/ \
                                        --env http_proxy=${http_proxy} \
                                        --env https_proxy=${http_proxy} \
                                        --env no_proxy=.intel.com \
                                        --env work_space=${WORKSPACE} \
                                    "){
                                        sh'''
                                            #!/bin/bash
                                            set -x 

                                            export log_dir=${WORKSPACE}/logs

                                            if [ "$request_rate" = "default" ];then
                                                apt install jq -y
                                                request_config=`jq --arg h ${hardware} --arg m ${modelid} --arg p ${dtype} --arg t ${tp_socket} --arg c ${length_config} \'.["vllm"][$h][$m][$p][$t][$c]["batch_size"]\' ${WORKSPACE}/vllm_scripts/tp_batchsize.json |sed 's/"//g'`
						                        echo "request_config is " $request_config

                                                reference_thp=`jq --arg h ${hardware} --arg m ${modelid} --arg p ${dtype} --arg t ${tp_socket} --arg c ${length_config} \'.["vllm"][$h][$m][$p][$t][$c]["baseline_thp"]\' ${WORKSPACE}/vllm_scripts/tp_batchsize.json |sed 's/"//g'`
                                                reference_first_token=`jq --arg h ${hardware} --arg m ${modelid} --arg p ${dtype} --arg t ${tp_socket} --arg c ${length_config} \'.["vllm"][$h][$m][$p][$t][$c]["baseline_first_token"]\' ${WORKSPACE}/vllm_scripts/tp_batchsize.json |sed 's/"//g'`
                                                reference_next_token=`jq --arg h ${hardware} --arg m ${modelid} --arg p ${dtype} --arg t ${tp_socket} --arg c ${length_config} \'.["vllm"][$h][$m][$p][$t][$c]["baseline_next_token"]\' ${WORKSPACE}/vllm_scripts/tp_batchsize.json |sed 's/"//g'`
                                                
                                            fi

                                            if [  "$benchmark_script" = "llmperf_serving" ];then
                                                echo 'Benchmark running with llmperf....'
                                                client_scripts="vllm_loadbalencer.sh"
                                                export request_rate=${request_config}
                                                echo Baseline_$modelid,$benchmark_script,$dataset,${tp_socket},$length_config,${num_prompt},$request_rate,$reference_thp,$reference_first_token,$reference_next_token| tee -a ${log_dir}/summary.log
                                            else
                                                client_scripts="vllm_benchmark.sh"

                                                if [ "$request_rate" = "default" ];then
                                                    echo 'Benchmark running with vllm default benchmark....'
                                                    export num_prompt=${request_config}
                                                    export request_rate=inf
                                                else
                                                    echo 'Benchmark running with vllm benchmark....'
                                                fi

                                                echo Baseline_$modelid,$benchmark_script,$dataset,${tp_socket},$length_config,${num_prompt},$request_rate,$reference_thp,$reference_first_token,$reference_next_token| tee -a ${log_dir}/summary.log
                                            fi
                                            
                                            bash ${client_scripts}
                                        '''
                                    }//docker
                                }//with env inside
                            }//for loop: request_rates
                        }//for loop: num prompts
                    } //length_config
                withEnv(["modelid=${modelid}","dtype=${dtype}"]){
                    sh '''
                        if [ "$benchmark_script" =~ "serving" ];then
                            container_name=vllm-$dtype 
                            docker stop ${container_name}
                        fi
                    '''
                    }
            }//loop: dtype
        }// for loop: model ids
    if (fileExists("${WORKSPACE}/logs/summary.log") == true){
                archiveArtifacts "logs/summary.log"
    }   
    archiveArtifacts artifacts: "**/logs/**", excludes: null
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
