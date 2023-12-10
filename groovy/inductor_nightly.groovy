ICX_NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('ICX_NODE_LABEL' in params) {
    echo "ICX_NODE_LABEL in params"
    if (params.ICX_NODE_LABEL != '') {
        ICX_NODE_LABEL = params.ICX_NODE_LABEL
    }
}
echo "ICX_NODE_LABEL: $ICX_NODE_LABEL"

SPR_NODE_LABEL = 'mlp-spr-04.sh.intel.com'
if ('SPR_NODE_LABEL' in params) {
    echo "SPR_NODE_LABEL in params"
    if (params.SPR_NODE_LABEL != '') {
        SPR_NODE_LABEL = params.SPR_NODE_LABEL
    }
}
echo "SPR_NODE_LABEL: $SPR_NODE_LABEL"

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

GNNBench = 'false'
if ('GNNBench' in params) {
    echo "GNNBench in params"
    if (params.GNNBench != '') {
        GNNBench = params.GNNBench
    }
}
echo "GNNBench: $GNNBench"

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

PRECISION = 'float32'
if ('PRECISION' in params) {
    echo "PRECISION in params"
    if (params.PRECISION != '') {
        PRECISION = params.PRECISION
    }
}
echo "PRECISION: $PRECISION"

TEST_SHAPE = 'static'
if ('TEST_SHAPE' in params) {
    echo "TEST_SHAPE in params"
    if (params.TEST_SHAPE != '') {
        TEST_SHAPE = params.TEST_SHAPE
    }
}
echo "TEST_SHAPE: $TEST_SHAPE"

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

image_tag = 'nightly'
if ('image_tag' in params) {
    echo "image_tag in params"
    if (params.image_tag != '') {
        image_tag = params.image_tag
    }
}
echo "image_tag: $image_tag"

// set reference build
refer_build = '0'
if( 'refer_build' in params && params.refer_build != '' ) {
    refer_build = params.refer_build
}
echo "refer_build: $refer_build"

HF_TOKEN= 'hf_xx'
if ('HF_TOKEN' in params) {
    echo "HF_TOKEN in params"
    if (params.HF_TOKEN != '') {
        HF_TOKEN = params.HF_TOKEN
    }
}
echo "HF_TOKEN: $HF_TOKEN"


def get_time(){
    return new Date().format('yyyy-MM-dd')
}
env._VERSION = get_time()
println(env._VERSION)

env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env.ICX24_HF_CACHE = '/home/torch/huggingface'
env.SPR04_HF_CACHE = '/home2/yudongsi/spr-04/huggingface'
env._ICX_NODE = "$ICX_NODE_LABEL"
env._SPR_NODE = "$SPR_NODE_LABEL"
env._image_tag = "$image_tag"
env._HF_TOKEN = "$HF_TOKEN"

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

def llm_benchmark(node){
    withEnv(["exec_node=${node}"]){
        sh '''
        #!/usr/bin/env bash
        old_container=`docker ps |grep $USER |awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi
        if [ ${exec_node} == ${_SPR_NODE} ];then
            hf_cache=${SPR04_HF_CACHE}
            PRECISION=bfloat16
        else
            hf_cache=${ICX24_HF_CACHE}
            PRECISION=float32
        fi
        docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${hf_cache}:/workspace/huggingface -v ${WORKSPACE}/llm_bench_${exec_node}:/workspace/pytorch/llm_bench_${exec_node} ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
        docker cp scripts/llmbench/env_prepare.sh $USER:/workspace/pytorch
        docker cp scripts/llmbench/token_latency.patch $USER:/workspace
        docker cp scripts/llmbench/run_dynamo_llm.py $USER:/workspace/pytorch
        docker cp scripts/llmbench/generate_report.py $USER:/workspace/pytorch/llm_bench_${exec_node}
        docker exec -i $USER bash -c "bash env_prepare.sh ${PRECISION} llm_bench_${exec_node}"
        ''' 
        try{
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    filter: "llm_bench_${exec_node}/*.txt",
                    fingerprintArtifacts: true,
                    target: "refer/")
            }
        }catch(err){
            echo err.getMessage()
        }
        sh '''
        #!/usr/bin/env bash
        set +e
        docker cp refer/llm_bench_${exec_node} $USER:/workspace/pytorch/llm_bench_${exec_node} && sudo rm -rf refer/*
        docker exec -i $USER bash -c "cd llm_bench_${exec_node};python3 generate_report.py --url ${BUILD_URL} --node ${exec_node};rm -rf llm_bench_${exec_node};rm generate_report.py"
        '''              
    }
}

def report(node){
    if ("${GNNBench}" == "true") {
        try{
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    filter: 'gnn_bench/*.txt',
                    fingerprintArtifacts: true,
                    target: "gnn_bench/")
            }
        }catch(err){
            echo err.getMessage()
        }
        sh '''
        #!/usr/bin/env bash
        cd ${WORKSPACE}/gnn_bench && python3 generate_report.py --url ${BUILD_URL} && rm -rf gnn_bench && rm generate_report.py 
        '''
    }//GNNBench_report       
}

def atfs(node){
    withEnv(["exec_node=${node}"]){
        if("${exec_node}" == "${_ICX_NODE}"){
            if ("${OPBench}" == "true"){
                archiveArtifacts artifacts: "**/opbench_log/**", fingerprint: true
            }
            if ("${ModelBench}" == "true"){
                archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
            }
            if ("${LLMBench}" == "true") {
                archiveArtifacts artifacts: "**/llm_bench_${exec_node}/**", fingerprint: true
            }
            if ("${GNNBench}" == "true") {
                archiveArtifacts artifacts: "**/gnn_bench/**", fingerprint: true
            }        
        }
        if("${exec_node}" == "${_SPR_NODE}"){
            if ("${LLMBench}" == "true") {
                archiveArtifacts artifacts: "**/llm_bench_${exec_node}/**", fingerprint: true
            }          
        }

    }
}

def mail_sent(node){
    if ("${debug}" == "true"){
        maillist="yudong.si@intel.com"
    }else{
        maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiangdong.zeng@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com;yanbing.jiang@intel.com"
    } 
    withEnv(["exec_node=${node}"]){
        if("${exec_node}" == "${_ICX_NODE}"){
            if ("${OPBench}" == "true"){
                if (fileExists("${WORKSPACE}/opbench_log/op-microbench-${_VERSION}.xlsx") == true){
                    emailext(
                        subject: "Torchinductor OP microbench Nightly Report ${_VERSION}",
                        mimeType: "text/html",
                        attachmentsPattern: "**/opbench_log/*.xlsx",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: '${FILE,path="opbench_log/ops.html"}'
                    )
                }else{
                    emailext(
                        subject: "Failure occurs in Torchinductor OP microbench Nightly ${_VERSION}",
                        mimeType: "text/html",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: 'Job build failed, please double check in ${BUILD_URL}'
                    )
                }
            }//OPBench
            if ("${ModelBench}" == "true"){
                if (fileExists("${WORKSPACE}/inductor_log/inductor_dashboard_regression_check.xlsx") == true){
                    emailext(
                        subject: "Torchinductor ModelBench Report ${_VERSION}",
                        mimeType: "text/html",
                        attachmentsPattern: "**/inductor_log/*.xlsx",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
                    )
                }else{
                    emailext(
                        subject: "Failure occurs in Torchinductor ModelBench ${_VERSION}",
                        mimeType: "text/html",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: 'Job build failed, please double check in ${BUILD_URL}'
                    )
                }
            }//ModelBench
            if ("${GNNBench}" == "true"){
                if (fileExists("${WORKSPACE}/gnn_bench/gnn_report.html") == true){
                    emailext(
                        subject: "Torchinductor GNNBench Report ${_VERSION}",
                        mimeType: "text/html",
                        attachmentsPattern: "**/gnn_bench/result.txt",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: '${FILE,path="gnn_bench/gnn_report.html"}'
                    )
                }else{
                    emailext(
                        subject: "Failure occurs in Torchinductor GNNBench ${_VERSION}",
                        mimeType: "text/html",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: 'Job build failed, please double check in ${BUILD_URL}'
                    )
                }
            }//GNNBench
            if ("${LLMBench}" == "true"){
                if (fileExists("${WORKSPACE}/llm_bench_${exec_node}/llm_report.html") == true){
                    emailext(
                        subject: "Torchinductor LLMBench Report ${_VERSION}",
                        mimeType: "text/html",
                        attachmentsPattern: "**/llm_bench_${exec_node}/result.txt",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: '${FILE,path="llm_bench_mlp-validate-icx24-ubuntu/llm_report.html"}'
                    )
                }else{
                    emailext(
                        subject: "Failure occurs in Torchinductor LLMBench ${_VERSION}",
                        mimeType: "text/html",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: 'Job build failed, please double check in ${BUILD_URL}'
                    )
                }
            }//LLMBench                           
        }
        if("${exec_node}" == "${_SPR_NODE}"){
            if ("${LLMBench}" == "true"){
                if (fileExists("${WORKSPACE}/llm_bench_${exec_node}/llm_report.html") == true){
                    emailext(
                        subject: "Torchinductor LLMBench Report ${_VERSION}",
                        mimeType: "text/html",
                        attachmentsPattern: "**/llm_bench_${exec_node}/result.txt",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: '${FILE,path="llm_bench_mlp-spr-04.sh.intel.com/llm_report.html"}'
                    )
                }else{
                    emailext(
                        subject: "Failure occurs in Torchinductor LLMBench ${_VERSION}",
                        mimeType: "text/html",
                        from: "pytorch_inductor_val@intel.com",
                        to: maillist,
                        body: 'Job build failed, please double check in ${BUILD_URL}'
                    )
                }
            }//LLMBench
        }      
    }
}

stage('Benchmark') {
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
            [$class: 'StringParameterValue', name: 'HF_TOKEN', value: "${HF_TOKEN}"],
        ]
    }   
    parallel icx24: {
        node(ICX_NODE_LABEL){
            stage("prepare"){
                echo 'prepare......'
                cleanup()
                deleteDir()
                checkout scm
                withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                    sh '''
                    #!/usr/bin/env bash
                    old_container=`docker ps |grep $USER |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    old_image_id=`docker images|grep pt_inductor|grep ${_image_tag}|awk '{print $3}'`
                    old_image=`echo $old_image_id| awk '{print $1}'`
                    if [ -n "${old_image}" ]; then
                        docker rmi -f $old_image
                    fi
                    docker system prune -f
                    docker login ccr-registry.caas.intel.com -u $USERNAME -p $PASSWORD
                    docker pull ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                    ''' 
                }    
            }

            stage('OPBench') {
                if ("${OPBench}" == "true") {
                    echo 'OPBench......'
                    sh '''
                    #!/usr/bin/env bash
                    old_container=`docker ps |grep $USER |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log:/workspace/pytorch/dynamo_opbench ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                    docker cp scripts/microbench/microbench_parser.py $USER:/workspace/pytorch
                    docker cp scripts/microbench/microbench.sh $USER:/workspace/pytorch
                    docker exec $USER bash -c "bash microbench.sh dynamo_opbench timm ${op_repeats} ${PRECISION}"
                    docker exec $USER bash -c "bash microbench.sh dynamo_opbench torchbench ${op_repeats} ${PRECISION}"
                    docker exec $USER bash -c "bash microbench.sh dynamo_opbench huggingface ${op_repeats} ${PRECISION}"
                    docker exec $USER bash -c "cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py -o ${_VERSION} -l ${BUILD_URL} -n ${_ICX_NODE};rm microbench_parser.py"
                    '''
                }
            }//OPBench

            stage("ModelBench"){
                if ("${ModelBench}" == "true") {
                    echo 'ModelBench......'
                    sh '''
                    #!/usr/bin/env bash
                    old_container=`docker ps |grep $USER |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi            
                    docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/torch/.cache:/root/.cache -v ${WORKSPACE}/inductor_log:/workspace/pytorch/inductor_log ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                    docker cp scripts/modelbench/inductor_test.sh $USER:/workspace/pytorch
                    docker cp scripts/modelbench/log_parser.py $USER:/workspace/pytorch
                    docker exec -i $USER bash -c "bash inductor_test.sh ${THREADS} ${CHANNELS} ${PRECISION} ${TEST_SHAPE} inductor_log ${WRAPPER} ${HF_TOKEN} inductor inference $SUITE $EXTRA;pip install styleframe;python log_parser.py --target inductor_log -m ${THREAD};cp inductor_dashboard_regression_check.xlsx inductor_log;cp inductor_model_bench.html inductor_log"
                    '''
                }
            }//ModelBench

            stage("LLMBench"){
                if ("${LLMBench}" == "true") {
                    llm_benchmark(ICX_NODE_LABEL)
                }
            }//LLMBench

            stage("GNNBench"){
                if ("${GNNBench}" == "true") {
                echo 'GNNBench......'
                sh '''
                #!/usr/bin/env bash
                old_container=`docker ps |grep $USER |awk '{print $1}'`
                if [ -n "${old_container}" ]; then
                    docker stop $old_container
                    docker rm $old_container
                    docker container prune -f
                fi
                docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v /home/torch/dataset:/workspace/dataset -v ${WORKSPACE}/gnn_bench:/workspace/gnn_bench ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                docker cp scripts/gnnbench/gnn_bench.sh $USER:/workspace
                docker cp scripts/gnnbench/generate_report.py $USER:/workspace/gnn_bench
                #workaround for build failure when build pyg-lib outside container
                docker exec -i $USER bash -c "pip uninstall pyg-lib -y;pip install git+https://github.com/pyg-team/pyg-lib.git"
                docker exec -i $USER bash -c "cd /workspace;bash gnn_bench.sh gnn_bench"
                '''
                }
            }//GNNBench    
            
            stage("Generate Report") {
                report(ICX_NODE_LABEL)
            } 
            stage('archiveArtifacts') {
                atfs(ICX_NODE_LABEL)        
            }
            stage("Sent Email"){
                mail_sent(ICX_NODE_LABEL)                      
            } 
        }
    },
    spr04: {
        node(SPR_NODE_LABEL) {
            stage("prepare"){
                echo 'prepare......'
                cleanup()
                withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                    sh '''
                    #!/usr/bin/env bash
                    sudo rm -rf ${WORKSPACE}/*
                    old_container=`docker ps |grep $USER |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    old_image_id=`docker images|grep pt_inductor|grep ${_image_tag}|awk '{print $3}'`
                    old_image=`echo $old_image_id| awk '{print $1}'`
                    if [ -n "${old_image}" ]; then
                        docker rmi -f $old_image
                    fi
                    docker system prune -f
                    docker login ccr-registry.caas.intel.com -u $USERNAME -p $PASSWORD
                    docker pull ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                    '''
                }
                checkout scm                
            }
            stage("LLMBench"){
                if ("${LLMBench}" == "true") {
                    llm_benchmark(SPR_NODE_LABEL)
                }
            }//LLMBench
            stage('archiveArtifacts') {
                atfs(SPR_NODE_LABEL)       
            }
            stage("Sent Email"){
                mail_sent(SPR_NODE_LABEL)                       
            }                                     
        }
    }
}
