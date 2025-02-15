NODE_LABEL = 'images_build_inductor_dashboard'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

BASE_IMAGE = 'ubuntu:20.04'
if ('BASE_IMAGE' in params) {
    echo "BASE_IMAGE in params"
    if (params.BASE_IMAGE != '') {
        BASE_IMAGE = params.BASE_IMAGE
    }
}
echo "BASE_IMAGE: $BASE_IMAGE"

PT_REPO = 'https://github.com/pytorch/pytorch.git'
if ('PT_REPO' in params) {
    echo "PT_REPO in params"
    if (params.PT_REPO != '') {
        PT_REPO = params.PT_REPO
    }
}
echo "PT_REPO: $PT_REPO"

PT_COMMIT = 'nightly'
if ('PT_COMMIT' in params) {
    echo "PT_COMMIT in params"
    if (params.PT_COMMIT != '') {
        PT_COMMIT = params.PT_COMMIT
    }
}
echo "PT_COMMIT: $PT_COMMIT"

TORCH_VISION_COMMIT = 'nightly'
if ('TORCH_VISION_COMMIT' in params) {
    echo "TORCH_VISION_COMMIT in params"
    if (params.TORCH_VISION_COMMIT != '') {
        TORCH_VISION_COMMIT = params.TORCH_VISION_COMMIT
    }
}
echo "TORCH_VISION_COMMIT: $TORCH_VISION_COMMIT"

TORCH_TEXT_COMMIT = 'nightly'
if ('TORCH_TEXT_COMMIT' in params) {
    echo "TORCH_TEXT_COMMIT in params"
    if (params.TORCH_TEXT_COMMIT != '') {
        TORCH_TEXT_COMMIT = params.TORCH_TEXT_COMMIT
    }
}
echo "TORCH_TEXT_COMMIT: $TORCH_TEXT_COMMIT"

TORCH_AUDIO_COMMIT = 'nightly'
if ('TORCH_AUDIO_COMMIT' in params) {
    echo "TORCH_AUDIO_COMMIT in params"
    if (params.TORCH_AUDIO_COMMIT != '') {
        TORCH_AUDIO_COMMIT = params.TORCH_AUDIO_COMMIT
    }
}
echo "TORCH_AUDIO_COMMIT: $TORCH_AUDIO_COMMIT"

TORCH_BENCH_COMMIT = 'main'
if ('TORCH_BENCH_COMMIT' in params) {
    echo "TORCH_BENCH_COMMIT in params"
    if (params.TORCH_BENCH_COMMIT != '') {
        TORCH_BENCH_COMMIT = params.TORCH_BENCH_COMMIT
    }
}
echo "TORCH_BENCH_COMMIT: $TORCH_BENCH_COMMIT"

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

tag = 'test'
if ('tag' in params) {
    echo "tag in params"
    if (params.tag != '') {
        tag = params.tag
    }
}
echo "tag: $tag"

HF_TOKEN= 'hf_xx'
if ('HF_TOKEN' in params) {
    echo "HF_TOKEN in params"
    if (params.HF_TOKEN != '') {
        HF_TOKEN = params.HF_TOKEN
    }
}
echo "HF_TOKEN: $HF_TOKEN"

iap_credential= 'chuanqiw_intel_id'
if ('iap_credential' in params) {
    echo "iap_credential in params"
    if (params.iap_credential != '') {
        iap_credential = params.iap_credential
    }
}
echo "iap_credential: $iap_credential"

driver_name= 'hotfix_agama-ci-devel-803.103'
if ('driver_name' in params) {
    echo "driver_name in params"
    if (params.driver_name != '') {
        driver_name = params.driver_name
    }
}
echo "iap_credential: $iap_credential"

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
            docker_img_status=`docker manifest inspect gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device}` || true
            if [ "${device}" = "cpu" ];then
                dockerfile="Dockerfile.cpu"
            elif [ "${device}" = "xpu" ];then
                dockerfile=Dockerfile.xpu
            elif [ "${device}" = "a100" ];then
                dockerfile=Dockerfile
            fi
            if [ -z "${docker_img_status}" ];then
                cp docker/${dockerfile} ./
                if [ "${device}" = "cpu" ];then
                    DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg BASE_IMAGE=${BASE_IMAGE} -t gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device} -f ${dockerfile} --target image .
                elif [ "${device}" = "xpu" ];then
                    DOCKER_BUILDKIT=1 docker build --no-cache --build-arg driver_name=${driver_name} \
                             -t gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device} -f Dockerfile.xpu .
                fi
            else
                echo "gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device} existed, skip build image"
            fi
            '''
        }
    }

    stage('push image') {
        retry(3){
            echo 'push image......'
            sh '''
            #!/usr/bin/env bash
            docker_img_status=`docker manifest inspect gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device}` || true
            if [ -z "${docker_img_status}" ];then
                docker push gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device}
            else
                echo "gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device} existed, skip push image"
            fi
            '''
        }
    }

    stage('clean image') {
        sh '''
        #!/usr/bin/env bash
        docker rmi -f gar-registry.caas.intel.com/pytorch/torchchat:${tag}_${device}
        '''
    }
    
}