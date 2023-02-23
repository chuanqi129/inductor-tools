NODE_LABEL = 'images_build_inductor_dashboard'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

node(NODE_LABEL){
    stage("get scripts and target image") {
        echo 'get scripts and target image......'
        sh '''
        #!/usr/bin/env bash
        cd ${WORKSPACE}
        rm -rf tmp
        git clone https://github.com/chuanqi129/inductor-tools.git tmp
        old_container=`docker ps |grep inductor_${target_tag}|awk '{print $1}'`
        if [ -n "${old_container}" ]; then
            docker stop $old_container
            docker rm $old_container
            docker container prune -f
        fi
        docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        '''
    }   
    stage("loggin container & cp scripts & collect graph"){
        echo 'running......'
        sh '''
        #!/usr/bin/env bash
        docker run -tid --name inductor_${target_tag} --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/graph/${target_tag}:/workspace/pytorch/${target_tag} ccr-registry.caas.intel.com/pytorch/pt_inductor:${target_tag}
        echo '#!/usr/bin/bash
        # set -x
        SUITE=$1
        MODEL=$2
        CHANNELS=$3
        BS=$4
        TARGET_TAG=$5

        MODEL_LIST=($(echo "${MODEL}" |sed '"'"'s/,/ /g'"'"'))

        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            bash inductor_cosim.sh ${SUITE} ${SINGLE_MODEL} ${CHANNELS} ${BS} ${TARGET_TAG}
        done' > run.sh
        docker cp run.sh inductor_${target_tag}:/workspace/pytorch
        docker cp tmp/scripts/cosim/inductor_cosim.sh inductor_${target_tag}:/workspace/pytorch
        docker cp tmp/scripts/cosim/inductor_cosim.py inductor_${target_tag}:/workspace/pytorch
        docker cp tmp/scripts/cosim/inductor_single_run.sh inductor_${target_tag}:/workspace/pytorch
        docker exec -i inductor_${target_tag} bash -c "bash run.sh ${suite} ${model} ${channels} ${bs} ${target_tag}"
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
        echo '#!/usr/bin/bash
        # set -x
        SUITE=$1
        MODEL=$2
        CHANNELS=$3
        BS=$4
        REFERENCE_TAG=$5

        MODEL_LIST=($(echo "${MODEL}" |sed '"'"'s/,/ /g'"'"'))

        for SINGLE_MODEL in ${MODEL_LIST[@]}
        do
            bash inductor_cosim.sh ${SUITE} ${SINGLE_MODEL} ${CHANNELS} ${BS} ${REFERENCE_TAG}
        done' > run.sh
        docker cp run.sh inductor_${reference_tag}:/workspace/pytorch        
        docker cp tmp/scripts/cosim/inductor_cosim.sh inductor_${reference_tag}:/workspace/pytorch
        docker cp tmp/scripts/cosim/inductor_cosim.py inductor_${reference_tag}:/workspace/pytorch
        docker cp tmp/scripts/cosim/inductor_single_run.sh inductor_${reference_tag}:/workspace/pytorch
        docker exec -i inductor_${reference_tag} bash -c "bash run.sh ${suite} ${model} ${channels} ${bs} ${reference_tag}"
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