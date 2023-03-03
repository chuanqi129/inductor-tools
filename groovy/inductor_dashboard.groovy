NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

debug = ''
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

Build_Image = ''
if ('Build_Image' in params) {
    echo "Build_Image in params"
    if (params.Build_Image != '') {
        Build_Image = params.Build_Image
    }
}
echo "Build_Image: $Build_Image"

isOP = ''
if ('isOP' in params) {
    echo "isOP in params"
    if (params.isOP != '') {
        isOP = params.isOP
    }
}
echo "isOP: $isOP"

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

op_suite = ''
if ('op_suite' in params) {
    echo "op_suite in params"
    if (params.op_suite != '') {
        op_suite = params.op_suite
    }
}
echo "op_suite: $op_suite"

op_repeats = ''
if ('op_repeats' in params) {
    echo "op_repeats in params"
    if (params.op_repeats != '') {
        op_repeats = params.op_repeats
    }
}
echo "op_repeats: $op_repeats"

THREAD = ''
if ('THREAD' in params) {
    echo "THREAD in params"
    if (params.THREAD != '') {
        THREAD = params.THREAD
    }
}
echo "THREAD: $THREAD"

CHANNELS = ''
if ('CHANNELS' in params) {
    echo "CHANNELS in params"
    if (params.CHANNELS != '') {
        CHANNELS = params.CHANNELS
    }
}
echo "CHANNELS: $CHANNELS"

MODEL_SUITE = ''
if ('MODEL_SUITE' in params) {
    echo "MODEL_SUITE in params"
    if (params.MODEL_SUITE != '') {
        MODEL_SUITE = params.MODEL_SUITE
    }
}
echo "MODEL_SUITE: $MODEL_SUITE"

inductor_tools_branch = ''
if ('inductor_tools_branch' in params) {
    echo "inductor_tools_branch in params"
    if (params.inductor_tools_branch != '') {
        inductor_tools_branch = params.inductor_tools_branch
    }
}
echo "inductor_tools_branch: $inductor_tools_branch"

def get_time(){
    return new Date().format('yyyy-MM-dd')
}
env._VERSION = get_time()
println(env._VERSION)

env._NODE = "$NODE_LABEL"

node(NODE_LABEL){
    checkout scm
    stage("get image and inductor-tools repo"){
        echo 'get image and inductor-tools repo......'
        if ("${Build_Image}" == "true") {
            def image_build_job = build job: 'inductor_images', propagate: false, parameters: [
                [$class: 'StringParameterValue', name: 'NODE_LABEL', value: 'Docker'],
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
                [$class: 'StringParameterValue', name: 'inductor_tools_branch', value: "${inductor_tools_branch}"],
                [$class: 'StringParameterValue', name: 'tag', value: 'nightly'],
            ]
            task_status = image_build_job.result
            task_number = image_build_job.number
            withEnv(["task_status=${task_status}","task_number=${task_number}"]) {
                sh '''#!/bin/bash
                    old_container=`docker ps |grep pt_inductor:nightly |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    old_image_id=`docker images|grep pt_inductor|grep nightly|awk '{print $3}'`
                    old_image=`echo $old_image_id| awk '{print $1}'`
                    if [ -n "${old_image}" ]; then
                        docker rmi -f $old_image
                    fi
                    cd ${WORKSPACE}
                    deleteDir()
                    checkout([
                        $class: 'GitSCM', 
                        branches: [[name: ${inductor_tools_branch}]],
                        userRemoteConfigs: [
                            [url: 'https://github.com/chuanqi129/inductor-tools.git']
                            ]
                    ])
                    if [ ${task_status} == "SUCCESS" ]; then
                        docker login ccr-registry.caas.intel.com
                        docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
                    fi
                '''
            }           
        }else {
            sh '''
            #!/usr/bin/env bash
            old_container=`docker ps |grep pt_inductor:nightly |awk '{print $1}'`
            if [ -n "${old_container}" ]; then
                docker stop $old_container
                docker rm $old_container
                docker container prune -f
            fi
            old_image_id=`docker images|grep pt_inductor|grep nightly|awk '{print $3}'`
            old_image=`echo $old_image_id| awk '{print $1}'`
            if [ -n "${old_image}" ]; then
                docker rmi -f $old_image
            fi
            cd ${WORKSPACE}
            deleteDir()            
            checkout([
                $class: 'GitSCM', 
                branches: [[name: ${inductor_tools_branch}]],
                userRemoteConfigs: [
                    [url: 'https://github.com/chuanqi129/inductor-tools.git']
                    ]
            ])
            mv inductor-tools/docker/Dockerfile ./
            DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg PT_REPO=${PT_REPO} --build-arg PT_BRANCH=${PT_BRANCH} --build-arg PT_COMMIT=${PT_COMMIT} --build-arg TORCH_VISION_BRANCH=${TORCH_VISION_BRANCH} --build-arg TORCH_VISION_COMMIT=${TORCH_VISION_COMMIT} --build-arg TORCH_DATA_BRANCH=${TORCH_DATA_BRANCH} --build-arg TORCH_DATA_COMMIT=${TORCH_DATA_COMMIT} --build-arg TORCH_TEXT_BRANCH=${TORCH_TEXT_BRANCH} --build-arg TORCH_TEXT_COMMIT=${TORCH_TEXT_COMMIT} --build-arg TORCH_AUDIO_BRANCH=${TORCH_AUDIO_BRANCH} --build-arg TORCH_AUDIO_COMMIT=${TORCH_AUDIO_COMMIT} --build-arg TORCH_BENCH_BRANCH=${TORCH_BENCH_BRANCH} --build-arg TORCH_BENCH_COMMIT=${TORCH_BENCH_COMMIT} --build-arg BENCH_COMMIT=${BENCH_COMMIT} -t ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly -f Dockerfile --target image .
            '''
        } 
    }
    stage('login container & prepare scripts & run') {
        echo 'login container & prepare scripts & run......'

        if ("${isOP}" == "true") {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name op_pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log:/workspace/pytorch/dynamo_opbench ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker cp tmp/scripts/microbench/microbench_parser.py op_pt_inductor:/workspace/pytorch
            docker cp tmp/scripts/microbench/microbench.sh op_pt_inductor:/workspace/pytorch
            docker exec -i op_pt_inductor bash -c "bash microbench.sh dynamo_opbench ${op_suite} ${op_repeats};cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py -w ${_VERSION} -l ${BUILD_URL} -n ${_NODE};rm microbench_parser.py"
            '''
        }else {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/inductor_log:/workspace/pytorch/inductor_log ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker cp tmp/scripts/modelbench/inductor_test.sh pt_inductor:/workspace/pytorch         
            docker cp tmp/scripts/modelbench/log_parser.py pt_inductor:/workspace/pytorch           
            docker exec -i pt_inductor bash inductor_test.sh ${THREAD} ${CHANNELS} inductor_log ${MODEL_SUITE}
            '''
        }
    }
    stage('archiveArtifacts') {
        if ("${isOP}" == "true"){
            archiveArtifacts artifacts: "**/opbench_log/**", fingerprint: true
        }else{
            archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
        }
    }
    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/opbench_log/op-microbench-${_VERSION}.xlsx") == true){
            emailext(
                subject: "Torchinductor OP microbench Nightly Report",
                mimeType: "text/html",
                attachmentsPattern: "**/opbench_log/*.xlsx",
                from: "Inductor_op_microbench_nightly@intel.com",
                to: maillist,
                body: '${FILE,path="opbench_log/ops.html"}'
            )
        }
        else{
            emailext(
                subject: "Failure occurs in Torchinductor OP microbench Nightly",
                mimeType: "text/html",
                from: "Inductor_op_microbench_nightly@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    } 
}