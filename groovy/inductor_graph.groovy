NODE_LABEL = 'mlp-validate-icx24-ubuntu'
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

channels = ''
if ('channels' in params) {
    echo "channels in params"
    if (params.channels != '') {
        channels = params.channels
    }
}
echo "channels: $channels"

suite = ''
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

bs = ''
if ('bs' in params) {
    echo "bs in params"
    if (params.bs != '') {
        bs = params.bs
    }
}
echo "bs: $bs"

inductor_tools_branch = ''
if ('inductor_tools_branch' in params) {
    echo "inductor_tools_branch in params"
    if (params.inductor_tools_branch != '') {
        inductor_tools_branch = params.inductor_tools_branch
    }
}
echo "inductor_tools_branch: $inductor_tools_branch"

node(NODE_LABEL){
    stage("get scripts and target image") {
        deleteDir()
        checkout scm       
        echo 'get scripts and target image......'
        sh '''
        #!/usr/bin/env bash
        old_container=`docker ps |grep inductor_${target_tag}|awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi
        docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        '''
    }   
    stage("loggin container & prepare scripts & collect graph"){
        echo 'running......'
        sh '''
        #!/usr/bin/env bash
        docker run -tid --name inductor_${target_tag} --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/graph/${target_tag}:/workspace/pytorch/${target_tag} ccr-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        docker cp scripts/cosim/inductor_cosim.sh inductor_${target_tag}:/workspace/pytorch
        docker cp scripts/cosim/inductor_cosim.py inductor_${target_tag}:/workspace/pytorch
        docker cp scripts/cosim/inductor_single_run.sh inductor_${target_tag}:/workspace/pytorch
        MODEL_LIST=($(echo "${model}" |sed 's/,/ /g'))
        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            docker exec -i inductor_${target_tag} bash -c "bash inductor_cosim.sh ${suite} ${SINGLE_MODEL} ${channels} ${bs} ${target_tag}"
        done        
        exit
        '''
    }    

    stage('collect graph in reference image') {
        if ("${reference}" == "true") {
        sh '''
        #!/usr/bin/env bash
        old_container=`docker ps |grep inductor_${reference_tag}|awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi        
        docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        docker run -tid --name inductor_${reference_tag} --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/graph/${reference_tag}:/workspace/pytorch/${reference_tag} ccr-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        docker cp scripts/cosim/inductor_cosim.sh inductor_${reference_tag}:/workspace/pytorch
        docker cp scripts/cosim/inductor_cosim.py inductor_${reference_tag}:/workspace/pytorch
        docker cp scripts/cosim/inductor_single_run.sh inductor_${reference_tag}:/workspace/pytorch
        MODEL_LIST=($(echo "${model}" |sed 's/,/ /g'))
        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            docker exec -i inductor_${reference_tag} bash -c "bash inductor_cosim.sh ${suite} ${SINGLE_MODEL} ${channels} ${bs} ${reference_tag}"
        done
        exit
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
    
    stage("clean"){
        echo 'clean container and image......'
        sh '''
        #!/usr/bin/env bash
        docker stop inductor_${target_tag}
        docker rm inductor_${target_tag}
        docker rmi ccr-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        '''
        if ("${reference}" == "true") {
        sh '''
        #!/usr/bin/env bash
        docker stop inductor_${reference_tag}
        docker rm inductor_${reference_tag}
        docker rmi ccr-registry.caas.intel.com/pytorch/pt_inductor:${reference_tag}
        '''
        }        
    }    

}
