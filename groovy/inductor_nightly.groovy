NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

debug = 'false'
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

IMAGE_BUILD_NODE = 'Docker'
if ('IMAGE_BUILD_NODE' in params) {
    echo "IMAGE_BUILD_NODE in params"
    if (params.IMAGE_BUILD_NODE != '') {
        IMAGE_BUILD_NODE = params.IMAGE_BUILD_NODE
    }
}
echo "IMAGE_BUILD_NODE: $IMAGE_BUILD_NODE"

Build_Image = 'True'
if ('Build_Image' in params) {
    echo "Build_Image in params"
    if (params.Build_Image != '') {
        Build_Image = params.Build_Image
    }
}
echo "Build_Image: $Build_Image"

BASE_IMAGE = 'ubuntu:20.04'
if ('BASE_IMAGE' in params) {
    echo "BASE_IMAGE in params"
    if (params.BASE_IMAGE != '') {
        BASE_IMAGE = params.BASE_IMAGE
    }
}
echo "BASE_IMAGE: $BASE_IMAGE"

OPBench = 'false'
if ('OPBench' in params) {
    echo "OPBench in params"
    if (params.OPBench != '') {
        OPBench = params.OPBench
    }
}
echo "OPBench: $OPBench"

ModelBench = 'false'
if ('ModelBench' in params) {
    echo "ModelBench in params"
    if (params.ModelBench != '') {
        ModelBench = params.ModelBench
    }
}
echo "ModelBench: $ModelBench"

LLMBench = 'false'
if ('LLMBench' in params) {
    echo "LLMBench in params"
    if (params.LLMBench != '') {
        LLMBench = params.LLMBench
    }
}
echo "LLMBench: $LLMBench"

transformers = '4.24.0'
if ('transformers' in params) {
    echo "transformers in params"
    if (params.transformers != '') {
        transformers = params.transformers
    }
}
echo "transformers: $transformers"

PT_REPO = 'https://github.com/pytorch/pytorch.git'
if ('PT_REPO' in params) {
    echo "PT_REPO in params"
    if (params.PT_REPO != '') {
        PT_REPO = params.PT_REPO
    }
}
echo "PT_REPO: $PT_REPO"

PT_BRANCH = 'nightly'
if ('PT_BRANCH' in params) {
    echo "PT_BRANCH in params"
    if (params.PT_BRANCH != '') {
        PT_BRANCH = params.PT_BRANCH
    }
}
echo "PT_BRANCH: $PT_BRANCH"

PT_COMMIT = 'nightly'
if ('PT_COMMIT' in params) {
    echo "PT_COMMIT in params"
    if (params.PT_COMMIT != '') {
        PT_COMMIT = params.PT_COMMIT
    }
}
echo "PT_COMMIT: $PT_COMMIT"

TORCH_VISION_BRANCH = 'nightly'
if ('TORCH_VISION_BRANCH' in params) {
    echo "TORCH_VISION_BRANCH in params"
    if (params.TORCH_VISION_BRANCH != '') {
        TORCH_VISION_BRANCH = params.TORCH_VISION_BRANCH
    }
}
echo "TORCH_VISION_BRANCH: $TORCH_VISION_BRANCH"

TORCH_VISION_COMMIT = 'nightly'
if ('TORCH_VISION_COMMIT' in params) {
    echo "TORCH_VISION_COMMIT in params"
    if (params.TORCH_VISION_COMMIT != '') {
        TORCH_VISION_COMMIT = params.TORCH_VISION_COMMIT
    }
}
echo "TORCH_VISION_COMMIT: $TORCH_VISION_COMMIT"

TORCH_TEXT_BRANCH = 'nightly'
if ('TORCH_TEXT_BRANCH' in params) {
    echo "TORCH_TEXT_BRANCH in params"
    if (params.TORCH_TEXT_BRANCH != '') {
        TORCH_TEXT_BRANCH = params.TORCH_TEXT_BRANCH
    }
}
echo "TORCH_TEXT_BRANCH: $TORCH_TEXT_BRANCH"

TORCH_TEXT_COMMIT = 'nightly'
if ('TORCH_TEXT_COMMIT' in params) {
    echo "TORCH_TEXT_COMMIT in params"
    if (params.TORCH_TEXT_COMMIT != '') {
        TORCH_TEXT_COMMIT = params.TORCH_TEXT_COMMIT
    }
}
echo "TORCH_TEXT_COMMIT: $TORCH_TEXT_COMMIT"

TORCH_AUDIO_BRANCH = 'nightly'
if ('TORCH_AUDIO_BRANCH' in params) {
    echo "TORCH_AUDIO_BRANCH in params"
    if (params.TORCH_AUDIO_BRANCH != '') {
        TORCH_AUDIO_BRANCH = params.TORCH_AUDIO_BRANCH
    }
}
echo "TORCH_AUDIO_BRANCH: $TORCH_AUDIO_BRANCH"

TORCH_AUDIO_COMMIT = 'nightly'
if ('TORCH_AUDIO_COMMIT' in params) {
    echo "TORCH_AUDIO_COMMIT in params"
    if (params.TORCH_AUDIO_COMMIT != '') {
        TORCH_AUDIO_COMMIT = params.TORCH_AUDIO_COMMIT
    }
}
echo "TORCH_AUDIO_COMMIT: $TORCH_AUDIO_COMMIT"

TORCH_BENCH_BRANCH = 'main'
if ('TORCH_BENCH_BRANCH' in params) {
    echo "TORCH_BENCH_BRANCH in params"
    if (params.TORCH_BENCH_BRANCH != '') {
        TORCH_BENCH_BRANCH = params.TORCH_BENCH_BRANCH
    }
}
echo "TORCH_BENCH_BRANCH: $TORCH_BENCH_BRANCH"

TORCH_BENCH_COMMIT = 'main'
if ('TORCH_BENCH_COMMIT' in params) {
    echo "TORCH_BENCH_COMMIT in params"
    if (params.TORCH_BENCH_COMMIT != '') {
        TORCH_BENCH_COMMIT = params.TORCH_BENCH_COMMIT
    }
}
echo "TORCH_BENCH_COMMIT: $TORCH_BENCH_COMMIT"

TORCH_DATA_BRANCH = 'nightly'
if ('TORCH_DATA_BRANCH' in params) {
    echo "TORCH_DATA_BRANCH in params"
    if (params.TORCH_DATA_BRANCH != '') {
        TORCH_DATA_BRANCH = params.TORCH_DATA_BRANCH
    }
}
echo "TORCH_DATA_BRANCH: $TORCH_DATA_BRANCH"

TORCH_DATA_COMMIT = 'nightly'
if ('TORCH_DATA_COMMIT' in params) {
    echo "TORCH_DATA_COMMIT in params"
    if (params.TORCH_DATA_COMMIT != '') {
        TORCH_DATA_COMMIT = params.TORCH_DATA_COMMIT
    }
}
echo "TORCH_DATA_COMMIT: $TORCH_DATA_COMMIT"

BENCH_COMMIT = 'nightly'
if ('BENCH_COMMIT' in params) {
    echo "BENCH_COMMIT in params"
    if (params.BENCH_COMMIT != '') {
        BENCH_COMMIT = params.BENCH_COMMIT
    }
}
echo "BENCH_COMMIT: $BENCH_COMMIT"

op_suite = 'huggingface'
if ('op_suite' in params) {
    echo "op_suite in params"
    if (params.op_suite != '') {
        op_suite = params.op_suite
    }
}
echo "op_suite: $op_suite"

op_repeats = '30'
if ('op_repeats' in params) {
    echo "op_repeats in params"
    if (params.op_repeats != '') {
        op_repeats = params.op_repeats
    }
}
echo "op_repeats: $op_repeats"

DT = 'float32'
if ('DT' in params) {
    echo "DT in params"
    if (params.DT != '') {
        DT = params.DT
    }
}
echo "DT: $DT"

SHAPE = 'static'
if ('SHAPE' in params) {
    echo "SHAPE in params"
    if (params.SHAPE != '') {
        SHAPE = params.SHAPE
    }
}
echo "SHAPE: $SHAPE"

THREAD = 'all'
if ('THREAD' in params) {
    echo "THREAD in params"
    if (params.THREAD != '') {
        THREAD = params.THREAD
    }
}
echo "THREAD: $THREAD"

CHANNELS = 'first'
if ('CHANNELS' in params) {
    echo "CHANNELS in params"
    if (params.CHANNELS != '') {
        CHANNELS = params.CHANNELS
    }
}
echo "CHANNELS: $CHANNELS"

MODEL_SUITE = 'all'
if ('MODEL_SUITE' in params) {
    echo "MODEL_SUITE in params"
    if (params.MODEL_SUITE != '') {
        MODEL_SUITE = params.MODEL_SUITE
    }
}
echo "MODEL_SUITE: $MODEL_SUITE"

image_tag = 'test'
if ('image_tag' in params) {
    echo "image_tag in params"
    if (params.image_tag != '') {
        image_tag = params.image_tag
    }
}
echo "image_tag: $image_tag"

def get_time(){
    return new Date().format('yyyy-MM-dd')
}
env._VERSION = get_time()
println(env._VERSION)

env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env._NODE = "$NODE_LABEL"

def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        docker stop $(docker ps -a -q)
        docker container prune -f
        docker system prune -f
        '''
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

node(NODE_LABEL){
    stage("get image and inductor-tools repo"){
        echo 'get image and inductor-tools repo......'
        cleanup()
        deleteDir()
        checkout scm
        if ("${Build_Image}" == "true") {
            def image_build_job = build job: 'inductor_images', propagate: false, parameters: [
                [$class: 'StringParameterValue', name: 'NODE_LABEL', value: "${IMAGE_BUILD_NODE}"],
                [$class: 'StringParameterValue', name: 'BASE_IMAGE', value: "${BASE_IMAGE}"],                
                [$class: 'StringParameterValue', name: 'PT_REPO', value: "${PT_REPO}"],
                [$class: 'StringParameterValue', name: 'PT_BRANCH', value: "${PT_BRANCH}"],
                [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${PT_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_VISION_BRANCH', value: "${TORCH_VISION_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_VISION_COMMIT', value: "${TORCH_VISION_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_TEXT_BRANCH', value: "${TORCH_TEXT_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_TEXT_COMMIT', value: "${TORCH_TEXT_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_DATA_BRANCH', value: "${TORCH_DATA_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_DATA_COMMIT', value: "${TORCH_DATA_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_AUDIO_BRANCH', value: "${TORCH_AUDIO_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_AUDIO_COMMIT', value: "${TORCH_AUDIO_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_BENCH_BRANCH', value: "${TORCH_BENCH_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_BENCH_COMMIT', value: "${TORCH_BENCH_COMMIT}"],
                [$class: 'StringParameterValue', name: 'BENCH_COMMIT', value: "${BENCH_COMMIT}"],
                [$class: 'StringParameterValue', name: 'tag', value: "${image_tag}"],
            ]
            task_status = image_build_job.result
            task_number = image_build_job.number
            withEnv(["task_status=${task_status}","task_number=${task_number}"]) {
                sh '''#!/bin/bash
                    tag=${image_tag}
                    old_container=`docker ps |grep $USER |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    old_image_id=`docker images|grep pt_inductor|grep ${tag}|awk '{print $3}'`
                    old_image=`echo $old_image_id| awk '{print $1}'`
                    if [ -n "${old_image}" ]; then
                        docker rmi -f $old_image
                    fi
                    if [ ${task_status} == "SUCCESS" ]; then
                        docker system prune -f
                        docker login ccr-registry.caas.intel.com -u yudongsi -p 1996+SYD
                        docker pull ${DOCKER_IMAGE_NAMESPACE}:${tag}
                    fi
                '''
            }           
        }else {
            sh '''
            #!/usr/bin/env bash
            tag=${image_tag}
            old_container=`docker ps |grep $USER |awk '{print $1}'`
            if [ -n "${old_container}" ]; then
                docker stop $old_container
                docker rm $old_container
                docker container prune -f
            fi
            old_image_id=`docker images|grep pt_inductor|grep ${tag}|awk '{print $3}'`
            old_image=`echo $old_image_id| awk '{print $1}'`
            if [ -n "${old_image}" ]; then
                docker rmi -f $old_image
            fi
            docker system prune -f
            docker login ccr-registry.caas.intel.com -u yudongsi -p 1996+SYD
            docker pull ${DOCKER_IMAGE_NAMESPACE}:${tag}
            '''        
        }
    }
    stage('login container & prepare scripts & run') {
        echo 'login container & prepare scripts & run......'
        if ("${OPBench}" == "true") {
            sh '''
            #!/usr/bin/env bash
            tag=${image_tag}
            docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log:/workspace/pytorch/dynamo_opbench ${DOCKER_IMAGE_NAMESPACE}:${tag}
            docker cp scripts/microbench/microbench_parser.py $USER:/workspace/pytorch
            docker cp scripts/microbench/microbench.sh $USER:/workspace/pytorch
            docker exec -i $USER bash -c "bash microbench.sh dynamo_opbench ${op_suite} ${op_repeats} ${DT};cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py -o ${_VERSION} -l ${BUILD_URL} -n ${_NODE};rm microbench_parser.py"
            '''
        }
        if ("${ModelBench}" == "true") {
            sh '''
            #!/usr/bin/env bash
            tag=${image_tag}
            docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/torch/.cache/torch/hub/checkpoints/:/root/.cache/torch/hub/checkpoints -v ${WORKSPACE}/inductor_log:/workspace/pytorch/inductor_log -v ${WORKSPACE}/Inductor Dashboard Regression Check inductor_log.xlsx:/workspace/pytorch/Inductor Dashboard Regression Check inductor_log.xlsx ${DOCKER_IMAGE_NAMESPACE}:${tag}
            docker cp scripts/modelbench/inductor_test.sh $USER:/workspace/pytorch         
            docker cp scripts/modelbench/log_parser.py $USER:/workspace/pytorch           
            docker exec -i $USER bash -c "bash inductor_test.sh ${THREAD} ${CHANNELS} ${DT} ${SHAPE} inductor_log ${MODEL_SUITE};python log_parser.py --target inductor_log"
            '''
        }
    }
    stage("LLMBench"){
        retry(3){
            if ("${LLMBench}" == "true") {
            echo 'LLMBench......'
            sh '''
            #!/usr/bin/env bash
            tag=${image_tag}
            old_container=`docker ps |grep $USER |awk '{print $1}'`
            if [ -n "${old_container}" ]; then
                docker stop $old_container
                docker rm $old_container
                docker container prune -f
            fi
            docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/torch/huggingface:/workspace/huggingface -v ${WORKSPACE}/llm_bench:/workspace/pytorch/llm_bench ${DOCKER_IMAGE_NAMESPACE}:${tag}
            docker cp scripts/llmbench/env_prepare.sh $USER:/workspace/pytorch
            docker cp scripts/llmbench/run_dynamo_gptj.py $USER:/workspace/pytorch
            docker cp scripts/llmbench/generate_report.py $USER:/workspace/pytorch
            docker exec -i $USER bash -c "bash env_prepare.sh ${transformers} ${DT}"
            '''
            }
        }
    }
    stage('archiveArtifacts') {
        if ("${OPBench}" == "true"){
            archiveArtifacts artifacts: "**/opbench_log/**", fingerprint: true
        }
        if ("${ModelBench}" == "true"){
            archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
            archiveArtifacts artifacts: "**/Inductor Dashboard Regression Check inductor_log.xlsx", fingerprint: true
        }
        if ("${LLMBench}" == "true") {
            archiveArtifacts artifacts: "**/llm_bench/**", fingerprint: true
        }
    }
    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if ("${OPBench}" == "true"){
            if (fileExists("${WORKSPACE}/opbench_log/op-microbench-${_VERSION}.xlsx") == true){
                emailext(
                    subject: "Torchinductor OP microbench Nightly Report",
                    mimeType: "text/html",
                    attachmentsPattern: "**/opbench_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="opbench_log/ops.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor OP microbench Nightly",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//OPBench
        if ("${ModelBench}" == "true"){
            if (fileExists("${WORKSPACE}/Inductor Dashboard Regression Check inductor_log.xlsx") == true){
                emailext(
                    subject: "Torchinductor ModelBench Nightly Report",
                    mimeType: "text/html",
                    attachmentsPattern: "**/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'html generation to do'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor ModelBench Nightly",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//ModelBench
        if ("${LLMBench}" == "true"){
            if (fileExists("${WORKSPACE}/llm_bench/llm_report.html") == true){
                emailext(
                    subject: "Torchinductor LLMBench Report",
                    mimeType: "text/html",
                    attachmentsPattern: "**/llm_bench/result.txt",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="llm_bench/llm_report.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor LLMBench",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//LLMBench                 
    } 
}
