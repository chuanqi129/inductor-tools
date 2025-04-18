NODE_LABEL = 'mlp-validate-clx07-centos'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}

target_tag = ''
if ('target_tag' in params) {
    echo "target_tag in params"
    if (params.target_tag != '') {
        target_tag = params.target_tag
    }
}
echo "target_tag: $target_tag"

reference_tag = ''
if ('reference_tag' in params) {
    echo "reference_tag in params"
    if (params.reference_tag != '') {
        reference_tag = params.reference_tag
    }
}
echo "reference_tag: $reference_tag"

reference = ''
if ('reference' in params) {
    echo "reference in params"
    if (params.reference != '') {
        reference = params.reference
    }
}
echo "reference: $reference"

THREAD = 'multiple'
if ('THREAD' in params) {
    echo "THREAD in params"
    if (params.THREAD != '') {
        THREAD = params.THREAD
    }
}
echo "THREAD: $THREAD"

MODE = 'inference'
if ('MODE' in params) {
    echo "MODE in params"
    if (params.MODE != '') {
        MODE = params.MODE
    }
}
echo "MODE: $MODE"

SCENARIO = 'performance'
if ('SCENARIO' in params) {
    echo "SCENARIO in params"
    if (params.SCENARIO != '') {
        SCENARIO = params.SCENARIO
    }
}
echo "SCENARIO: $SCENARIO"

channels = 'first'
if ('channels' in params) {
    echo "channels in params"
    if (params.channels != '') {
        channels = params.channels
    }
}
echo "channels: $channels"

suite = 'torchbench'
if ('suite' in params) {
    echo "suite in params"
    if (params.suite != '') {
        suite = params.suite
    }
}
echo "suite: $suite"

model = ''
if ('model' in params) {
    echo "model in params"
    if (params.model != '') {
        model = params.model
    }
}
echo "model: $model"

bs = '0'
if ('bs' in params) {
    echo "bs in params"
    if (params.bs != '') {
        bs = params.bs
    }
}
echo "bs: $bs"

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

WRAPPER = 'default'
if ('WRAPPER' in params) {
    echo "WRAPPER in params"
    if (params.WRAPPER != '') {
        WRAPPER = params.WRAPPER
    }
}
echo "WRAPPER: $WRAPPER"

def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        docker container prune -f
        docker system prune -f
        sudo chmod 777  ${WORKSPACE}
        sudo rm -rf ${WORKSPACE}/graph/        
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
    stage("get scripts and target image") {
        cleanup()        
        deleteDir()
        checkout scm       
        echo 'get scripts and target image......'
        sh '''
        #!/usr/bin/env bash
        old_container=`docker ps |grep $USER |awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi
        docker pull gar-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        '''
    }   
    stage("loggin container & prepare scripts & collect graph"){
        echo 'running......'
        sh '''
        #!/usr/bin/env bash
        docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/graph/${target_tag}:/workspace/pytorch/${target_tag} gar-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        docker cp scripts/cosim/inductor_cosim.sh $USER:/workspace/pytorch
        docker cp scripts/cosim/inductor_cosim.py $USER:/workspace/pytorch
        docker cp scripts/cosim/inductor_single_run.sh $USER:/workspace/pytorch
        MODEL_LIST=($(echo "${model}" |sed 's/,/ /g'))
        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            docker exec -i $USER bash -c "bash inductor_cosim.sh ${THREAD} ${MODE} ${SCENARIO} ${suite} ${SINGLE_MODEL} ${DT} ${channels} ${SHAPE} ${WRAPPER} ${bs} ${target_tag}"
        done        
        exit
        docker rmi -f gar-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        '''
    }    

    stage('collect graph in reference image') {
        if ("${reference}" == "true") {
        sh '''
        #!/usr/bin/env bash
        old_container=`docker ps |grep $USER |awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi        
        docker pull gar-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/graph/${reference_tag}:/workspace/pytorch/${reference_tag} gar-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        docker cp scripts/cosim/inductor_cosim.sh $USER:/workspace/pytorch
        docker cp scripts/cosim/inductor_cosim.py $USER:/workspace/pytorch
        docker cp scripts/cosim/inductor_single_run.sh $USER:/workspace/pytorch
        MODEL_LIST=($(echo "${model}" |sed 's/,/ /g'))
        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            docker exec -i $USER bash -c "bash inductor_cosim.sh ${THREAD} ${MODE} ${SCENARIO} ${suite} ${SINGLE_MODEL} ${DT} ${channels} ${SHAPE} ${WRAPPER} ${bs} ${reference_tag}"
        done
        exit
        docker rmi -f gar-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        '''
        }else {
            echo 'skip collection in reference image'
        }
    }

    stage("artifacts files") {
        echo 'artifacts files......'
        archiveArtifacts artifacts: "**/graph/${target_tag}/**/graph.py", fingerprint: true
        if ("${reference}" == "true"){
            archiveArtifacts artifacts: "**/graph/${reference_tag}/**/graph.py", fingerprint: true
        }
    }

}