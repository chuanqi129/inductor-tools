NODE_LABEL = 'images_build_inductor_dashboard'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

BASE_IMAGE = ''
if ('BASE_IMAGE' in params) {
    echo "BASE_IMAGE in params"
    if (params.BASE_IMAGE != '') {
        BASE_IMAGE = params.BASE_IMAGE
    }
}
echo "BASE_IMAGE: $BASE_IMAGE"

PT_REPO = ''
if ('PT_REPO' in params) {
    echo "PT_REPO in params"
    if (params.PT_REPO != '') {
        PT_REPO = params.PT_REPO
    }
}
echo "PT_REPO: $PT_REPO"

PT_BRANCH = ''
if ('PT_BRANCH' in params) {
    echo "PT_BRANCH in params"
    if (params.PT_BRANCH != '') {
        PT_BRANCH = params.PT_BRANCH
    }
}
echo "PT_BRANCH: $PT_BRANCH"

PT_COMMIT = ''
if ('PT_COMMIT' in params) {
    echo "PT_COMMIT in params"
    if (params.PT_COMMIT != '') {
        PT_COMMIT = params.PT_COMMIT
    }
}
echo "PT_COMMIT: $PT_COMMIT"

TORCH_VISION_BRANCH = ''
if ('TORCH_VISION_BRANCH' in params) {
    echo "TORCH_VISION_BRANCH in params"
    if (params.TORCH_VISION_BRANCH != '') {
        TORCH_VISION_BRANCH = params.TORCH_VISION_BRANCH
    }
}
echo "TORCH_VISION_BRANCH: $TORCH_VISION_BRANCH"

TORCH_VISION_COMMIT = ''
if ('TORCH_VISION_COMMIT' in params) {
    echo "TORCH_VISION_COMMIT in params"
    if (params.TORCH_VISION_COMMIT != '') {
        TORCH_VISION_COMMIT = params.TORCH_VISION_COMMIT
    }
}
echo "TORCH_VISION_COMMIT: $TORCH_VISION_COMMIT"

TORCH_TEXT_BRANCH = ''
if ('TORCH_TEXT_BRANCH' in params) {
    echo "TORCH_TEXT_BRANCH in params"
    if (params.TORCH_TEXT_BRANCH != '') {
        TORCH_TEXT_BRANCH = params.TORCH_TEXT_BRANCH
    }
}
echo "TORCH_TEXT_BRANCH: $TORCH_TEXT_BRANCH"

TORCH_TEXT_COMMIT = ''
if ('TORCH_TEXT_COMMIT' in params) {
    echo "TORCH_TEXT_COMMIT in params"
    if (params.TORCH_TEXT_COMMIT != '') {
        TORCH_TEXT_COMMIT = params.TORCH_TEXT_COMMIT
    }
}
echo "TORCH_TEXT_COMMIT: $TORCH_TEXT_COMMIT"

TORCH_AUDIO_BRANCH = ''
if ('TORCH_AUDIO_BRANCH' in params) {
    echo "TORCH_AUDIO_BRANCH in params"
    if (params.TORCH_AUDIO_BRANCH != '') {
        TORCH_AUDIO_BRANCH = params.TORCH_AUDIO_BRANCH
    }
}
echo "TORCH_AUDIO_BRANCH: $TORCH_AUDIO_BRANCH"

TORCH_AUDIO_COMMIT = ''
if ('TORCH_AUDIO_COMMIT' in params) {
    echo "TORCH_AUDIO_COMMIT in params"
    if (params.TORCH_AUDIO_COMMIT != '') {
        TORCH_AUDIO_COMMIT = params.TORCH_AUDIO_COMMIT
    }
}
echo "TORCH_AUDIO_COMMIT: $TORCH_AUDIO_COMMIT"

TORCH_BENCH_BRANCH = ''
if ('TORCH_BENCH_BRANCH' in params) {
    echo "TORCH_BENCH_BRANCH in params"
    if (params.TORCH_BENCH_BRANCH != '') {
        TORCH_BENCH_BRANCH = params.TORCH_BENCH_BRANCH
    }
}
echo "TORCH_BENCH_BRANCH: $TORCH_BENCH_BRANCH"

TORCH_BENCH_COMMIT = ''
if ('TORCH_BENCH_COMMIT' in params) {
    echo "TORCH_BENCH_COMMIT in params"
    if (params.TORCH_BENCH_COMMIT != '') {
        TORCH_BENCH_COMMIT = params.TORCH_BENCH_COMMIT
    }
}
echo "TORCH_BENCH_COMMIT: $TORCH_BENCH_COMMIT"

TORCH_DATA_BRANCH = ''
if ('TORCH_DATA_BRANCH' in params) {
    echo "TORCH_DATA_BRANCH in params"
    if (params.TORCH_DATA_BRANCH != '') {
        TORCH_DATA_BRANCH = params.TORCH_DATA_BRANCH
    }
}
echo "TORCH_DATA_BRANCH: $TORCH_DATA_BRANCH"

TORCH_DATA_COMMIT = ''
if ('TORCH_DATA_COMMIT' in params) {
    echo "TORCH_DATA_COMMIT in params"
    if (params.TORCH_DATA_COMMIT != '') {
        TORCH_DATA_COMMIT = params.TORCH_DATA_COMMIT
    }
}
echo "TORCH_DATA_COMMIT: $TORCH_DATA_COMMIT"

BENCH_COMMIT = ''
if ('BENCH_COMMIT' in params) {
    echo "BENCH_COMMIT in params"
    if (params.BENCH_COMMIT != '') {
        BENCH_COMMIT = params.BENCH_COMMIT
    }
}
echo "BENCH_COMMIT: $BENCH_COMMIT"

inductor_tools_branch = ''
if ('inductor_tools_branch' in params) {
    echo "inductor_tools_branch in params"
    if (params.inductor_tools_branch != '') {
        inductor_tools_branch = params.inductor_tools_branch
    }
}
echo "inductor_tools_branch: $inductor_tools_branch"

tag = ''
if ('tag' in params) {
    echo "tag in params"
    if (params.tag != '') {
        tag = params.tag
    }
}
echo "tag: $tag"

node(NODE_LABEL){
    stage("get dockerfile"){
        echo 'get dockerfile......'
        deleteDir()
        checkout scm     
    }
    stage("nightly_pre") {
        if ("${tag}" == "nightly") {
            sh '''
            #!/usr/bin/env bash
            docker system prune -af
            docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker tag ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker push ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker rmi -f ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly_pre
            docker rmi -f ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            '''
        }else {
            sh '''
            #!/usr/bin/env bash
            docker system prune -af
            '''
        }
    }   
    stage("build image"){
        retry(3){
            echo 'Building image......'
            sh '''
            #!/usr/bin/env bash
            docker system prune -af
            mv docker/Dockerfile ./
            DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg PT_REPO=${PT_REPO} --build-arg PT_BRANCH=${PT_BRANCH} --build-arg PT_COMMIT=${PT_COMMIT} --build-arg TORCH_VISION_BRANCH=${TORCH_VISION_BRANCH} --build-arg TORCH_VISION_COMMIT=${TORCH_VISION_COMMIT} --build-arg TORCH_DATA_BRANCH=${TORCH_DATA_BRANCH} --build-arg TORCH_DATA_COMMIT=${TORCH_DATA_COMMIT} --build-arg TORCH_TEXT_BRANCH=${TORCH_TEXT_BRANCH} --build-arg TORCH_TEXT_COMMIT=${TORCH_TEXT_COMMIT} --build-arg TORCH_AUDIO_BRANCH=${TORCH_AUDIO_BRANCH} --build-arg TORCH_AUDIO_COMMIT=${TORCH_AUDIO_COMMIT} --build-arg TORCH_BENCH_BRANCH=${TORCH_BENCH_BRANCH} --build-arg TORCH_BENCH_COMMIT=${TORCH_BENCH_COMMIT} --build-arg BENCH_COMMIT=${BENCH_COMMIT} -t ccr-registry.caas.intel.com/pytorch/pt_inductor:${tag} -f Dockerfile --target image .
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
