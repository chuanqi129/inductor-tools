NODE_LABEL = 'images_build_inductor_dashboard'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

node(NODE_LABEL){
    stage("get dockerfile"){
        echo 'get dockerfile......'
        sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE}
            rm -rf tmp
            git clone -b ${inductor_tools_branch} https://github.com/chuanqi129/inductor-tools.git tmp
            mv tmp/docker/Dockerfile ./
            rm -rf tmp
        '''
    }
    stage("nightly_pre") {
        if ("${tag}" == "nightly") {
            sh '''
            #!/usr/bin/env bash
            docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker tag ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker login ccr-registry.caas.intel.com -u yudongsi -p 0608+SYD
            docker push ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker rmi -f ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker rmi -f ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            '''
        }
    }   
    stage("build image"){
        retry(3){
            echo 'Building image......'
            sh '''
            #!/usr/bin/env bash
            docker system prune -af
            DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg PT_REPO=${PT_REPO} --build-arg PT_BRANCH=${PT_BRANCH} --build-arg PT_COMMIT=${PT_COMMIT} --build-arg TORCH_VISION_BRANCH=${TORCH_VISION_BRANCH} --build-arg TORCH_VISION_COMMIT=${TORCH_VISION_COMMIT} --build-arg TORCH_DATA_BRANCH=${TORCH_DATA_BRANCH} --build-arg TORCH_DATA_COMMIT=${TORCH_DATA_COMMIT} --build-arg TORCH_TEXT_BRANCH=${TORCH_TEXT_BRANCH} --build-arg TORCH_TEXT_COMMIT=${TORCH_TEXT_COMMIT} --build-arg TORCH_AUDIO_BRANCH=${TORCH_AUDIO_BRANCH} --build-arg TORCH_AUDIO_COMMIT=${TORCH_AUDIO_COMMIT} --build-arg TORCH_BENCH_BRANCH=${TORCH_BENCH_BRANCH} --build-arg TORCH_BENCH_COMMIT=${TORCH_BENCH_COMMIT} --build-arg BENCH_COMMIT=${BENCH_COMMIT} -t ccr-registry.caas.intel.com/pytorch/pt_inductor:${tag} -f Dockerfile --target image .
            '''
        }
    }

    stage('push image') {
        echo 'push image......'
        sh '''
        #!/usr/bin/env bash
        docker push ccr-registry.caas.intel.com/pytorch/pt_inductor:${tag}
        docker rmi -f ccr-registry.caas.intel.com/pytorch/pt_inductor:${tag}
        '''
    }
}
