NODE_LABEL = 'mlp-spr-pytorch-dev1'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

debug = 'False'
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

debug_mail = 'yudong.si@intel.com'
if ('debug_mail' in params) {
    echo "debug_mail in params"
    if (params.debug_mail != '') {
        debug_mail = params.debug_mail
    }
}
echo "debug_mail: $debug_mail"

specify_image = 'false'
if ('specify_image' in params) {
    echo "specify_image in params"
    if (params.specify_image != '') {
        specify_image = params.specify_image
    }
}
echo "specify_image: $specify_image"

TORCH_REPO = 'https://github.com/pytorch/pytorch.git'
if ('TORCH_REPO' in params) {
    echo "TORCH_REPO in params"
    if (params.TORCH_REPO != '') {
        TORCH_REPO = params.TORCH_REPO
    }
}
echo "TORCH_REPO: $TORCH_REPO"

TORCH_BRANCH= 'nightly'
if ('TORCH_BRANCH' in params) {
    echo "TORCH_BRANCH in params"
    if (params.TORCH_BRANCH != '') {
        TORCH_BRANCH = params.TORCH_BRANCH
    }
}
echo "TORCH_BRANCH: $TORCH_BRANCH"

TORCH_COMMIT= 'nightly'
if ('TORCH_COMMIT' in params) {
    echo "TORCH_COMMIT in params"
    if (params.TORCH_COMMIT != '') {
        TORCH_COMMIT = params.TORCH_COMMIT
    }
}
echo "TORCH_COMMIT: $TORCH_COMMIT"

DYNAMO_BENCH= 'fea73cb'
if ('DYNAMO_BENCH' in params) {
    echo "DYNAMO_BENCH in params"
    if (params.DYNAMO_BENCH != '') {
        DYNAMO_BENCH = params.DYNAMO_BENCH
    }
}
echo "DYNAMO_BENCH: $DYNAMO_BENCH"

TORCH_AUDIO_BRANCH= 'nightly'
if ('TORCH_AUDIO_BRANCH' in params) {
    echo "TORCH_AUDIO_BRANCH in params"
    if (params.TORCH_AUDIO_BRANCH != '') {
        TORCH_AUDIO_BRANCH = params.TORCH_AUDIO_BRANCH
    }
}
echo "TORCH_AUDIO_BRANCH: $TORCH_AUDIO_BRANCH"

AUDIO= '0a652f5'
if ('AUDIO' in params) {
    echo "AUDIO in params"
    if (params.AUDIO != '') {
        AUDIO = params.AUDIO
    }
}
echo "AUDIO: $AUDIO"

TORCH_TEXT_BRANCH= 'nightly'
if ('TORCH_TEXT_BRANCH' in params) {
    echo "TORCH_TEXT_BRANCH in params"
    if (params.TORCH_TEXT_BRANCH != '') {
        TORCH_TEXT_BRANCH = params.TORCH_TEXT_BRANCH
    }
}
echo "TORCH_TEXT_BRANCH: $TORCH_TEXT_BRANCH"

TEXT= 'c4ad5dd'
if ('TEXT' in params) {
    echo "TEXT in params"
    if (params.TEXT != '') {
        TEXT = params.TEXT
    }
}
echo "TEXT: $TEXT"

TORCH_VISION_BRANCH= 'nightly'
if ('TORCH_VISION_BRANCH' in params) {
    echo "TORCH_VISION_BRANCH in params"
    if (params.TORCH_VISION_BRANCH != '') {
        TORCH_VISION_BRANCH = params.TORCH_VISION_BRANCH
    }
}
echo "TORCH_VISION_BRANCH: $TORCH_VISION_BRANCH"

VISION= 'f2009ab'
if ('VISION' in params) {
    echo "VISION in params"
    if (params.VISION != '') {
        VISION = params.VISION
    }
}
echo "VISION: $VISION"

TORCH_DATA_BRANCH= 'nightly'
if ('TORCH_DATA_BRANCH' in params) {
    echo "TORCH_DATA_BRANCH in params"
    if (params.TORCH_DATA_BRANCH != '') {
        TORCH_DATA_BRANCH = params.TORCH_DATA_BRANCH
    }
}
echo "TORCH_DATA_BRANCH: $TORCH_DATA_BRANCH"

DATA= '5cb3e6d'
if ('DATA' in params) {
    echo "DATA in params"
    if (params.DATA != '') {
        DATA = params.DATA
    }
}
echo "DATA: $DATA"

TORCH_BENCH_BRANCH= 'main'
if ('TORCH_BENCH_BRANCH' in params) {
    echo "TORCH_BENCH_BRANCH in params"
    if (params.TORCH_BENCH_BRANCH != '') {
        TORCH_BENCH_BRANCH = params.TORCH_BENCH_BRANCH
    }
}
echo "TORCH_BENCH_BRANCH: $TORCH_BENCH_BRANCH"

TORCH_BENCH= 'a0848e19'
if ('TORCH_BENCH' in params) {
    echo "TORCH_BENCH in params"
    if (params.TORCH_BENCH != '') {
        TORCH_BENCH = params.TORCH_BENCH
    }
}
echo "TORCH_BENCH: $TORCH_BENCH"

precision = 'float32'
if ('precision' in params) {
    echo "precision in params"
    if (params.precision != '') {
        precision = params.precision
    }
}
echo "precision: $precision"

test_mode = 'inference'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"

shape = 'static'
if ('shape' in params) {
    echo "shape in params"
    if (params.shape != '') {
        shape = params.shape
    }
}
echo "shape: $shape"

// set reference build
refer_build = ''
if( 'refer_build' in params && params.refer_build != '' ) {
    refer_build = params.refer_build
}
echo "refer_build: $refer_build"

gh_token = ''
if( 'gh_token' in params && params.gh_token != '' ) {
    gh_token = params.gh_token
}
echo "gh_token: $gh_token"

THREADS= 'all'
if ('THREADS' in params) {
    echo "THREADS in params"
    if (params.THREADS != '') {
        THREADS = params.THREADS
    }
}
echo "THREADS: $THREADS"

CHANNELS= 'first'
if ('CHANNELS' in params) {
    echo "CHANNELS in params"
    if (params.CHANNELS != '') {
        CHANNELS = params.CHANNELS
    }
}
echo "CHANNELS: $CHANNELS"

WRAPPER= 'default'
if ('WRAPPER' in params) {
    echo "WRAPPER in params"
    if (params.WRAPPER != '') {
        WRAPPER = params.WRAPPER
    }
}
echo "WRAPPER: $WRAPPER"

dash_board = 'false'
if( 'dash_board' in params && params.dash_board != '' ) {
    dash_board = params.dash_board
}
echo "dash_board: $dash_board"

dashboard_title = 'default'
if( 'dashboard_title' in params && params.dashboard_title != '' ) {
    dashboard_title = params.dashboard_title
}
echo "dashboard_title: $dashboard_title"

specify_image_tag = '2023_07_29_static_default_local'
if( 'specify_image_tag' in params && params.specify_image_tag != '' ) {
    specify_image_tag = params.specify_image_tag
}
echo "specify_image_tag: $specify_image_tag"

env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._gh_token = "$gh_token"
env._dash_board = "$dash_board"
env._dashboard_title = "$dashboard_title"
env._specify_image_tag = "$specify_image_tag"

env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH"

env._AUDIO = "$AUDIO"
env._TEXT = "$TEXT"
env._VISION = "$VISION"
env._DATA = "$DATA"
env._TORCH_BENCH = "$TORCH_BENCH"

env._THREADS = "$THREADS"
env._CHANNELS = "$CHANNELS"
env._WRAPPER = "$WRAPPER"
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env._image_tag = "${env._target}_${env._shape}_${env._WRAPPER}_local"

env.INDUCTOR_CACHE = '/home2/yudongsi/.cache'

def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        docker stop $(docker ps -a -q)
        docker container prune -f
        docker system prune -f
        cd ${WORKSPACE} && sudo rm -rf *
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
    stage("prepare"){
        echo 'prepare......'
        cleanup()
        deleteDir()
        checkout scm
    }
    stage("prepare container"){
        if ("${specify_image}" == "false"){
            withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                withCredentials([usernamePassword(credentialsId: 'syd_token_inteltf-jenk', usernameVariable: 'TG_USERNAME', passwordVariable: 'TG_PASSWORD')]){
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
                    
                    curl -s -I -k -u $TG_USERNAME:$TG_PASSWORD "https://inteltf-jenk.sh.intel.com/job/inductor_images/buildWithParameters?token=inductor_token&PT_REPO=`echo ${TORCH_REPO}`&PT_BRANCH=`echo ${TORCH_BRANCH}`&PT_COMMIT=`echo ${TORCH_COMMIT}`&TORCH_VISION_BRANCH=nightly&TORCH_VISION_COMMIT=`echo ${VISION}`&TORCH_TEXT_BRANCH=nightly&TORCH_TEXT_COMMIT=`echo ${TEXT}`&TORCH_DATA_BRANCH=nightly&TORCH_DATA_COMMIT=`echo ${DATA}`&TORCH_AUDIO_BRANCH=nightly&TORCH_AUDIO_COMMIT=`echo ${AUDIO}`&TORCH_BENCH_BRANCH=main&TORCH_BENCH_COMMIT=`echo ${TORCH_BENCH}`&BENCH_COMMIT=`echo ${DYNAMO_BENCH}`&tag=`echo ${_image_tag}`"
                    sleep 10s
                    
                    for t in {1..6}
                    do
                        build_result=$(curl -s -k -u $TG_USERNAME:$TG_PASSWORD "https://inteltf-jenk.sh.intel.com/job/inductor_images/lastBuild/api/json?pretty=true" | jq ".result")
                        if [ $build_result == '\"SUCCESS\"' ]; then
                            docker login ccr-registry.caas.intel.com -u $USERNAME -p $PASSWORD
                            docker pull ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
                            break
                        else
                            sleep 30m
                        fi
                    done
                    ''' 
                }
            }              
        }
        if ("${specify_image}" == "true")
        {
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
                docker pull ${DOCKER_IMAGE_NAMESPACE}:${_specify_image_tag}
                '''                 
            }          
        }
  
    }

    stage("benchmark") {
        sh '''
        #!/usr/bin/env bash
        if [ "${specify_image}" == "false" ]; then
             docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 1G -v ${INDUCTOR_CACHE}:/root/.cache -v ${WORKSPACE}/${_target}:/workspace/pytorch/${_target} ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
        else
             docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 1G -v ${INDUCTOR_CACHE}:/root/.cache -v ${WORKSPACE}/${_target}:/workspace/pytorch/${_target} ${DOCKER_IMAGE_NAMESPACE}:${_specify_image_tag}
        fi
        docker cp scripts/modelbench/inductor_test.sh $USER:/workspace/pytorch
        docker cp scripts/modelbench/inductor_train.sh $USER:/workspace/pytorch
        docker cp scripts/modelbench/report.py $USER:/workspace/pytorch
        docker exec -i $USER bash -c "export TRANSFORMERS_OFFLINE=1;bash inductor_test.sh ${_THREADS} ${_CHANNELS} ${_precision} ${_shape} inductor_log ${_DYNAMO_BENCH} ${_WRAPPER}"
        docker exec -i $USER bash -c "mv inductor_log ${_target}"

        '''
    }

    stage("generate report"){
        if ("${test_mode}" == "inference")
        {
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    fingerprintArtifacts: true
                )           
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && mkdir -p refer && mv inductor_log refer
                docker cp ${WORKSPACE}/refer $USER:/workspace/pytorch/
                if [ ${_dash_board} == "true" ]; then
                     docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -r refer -t ${_target} -m ${_THREADS} --gh_token ${_gh_token} --dashboard ${_dashboard_title} --url ${BUILD_URL}"
                else
                     docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -r refer -t ${_target} -m ${_THREADS} --md_off --precision ${_precision} --url ${BUILD_URL}"
                fi
                rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                if [ ${_dash_board} == "true" ]; then
                    docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -t ${_target} -m ${_THREADS} --gh_token ${_gh_token} --dashboard ${_dashboard_title} --precision ${_precision} --url ${BUILD_URL}"
                else
                    docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -t ${_target} -m ${_THREADS} --md_off --precision ${_precision} --url ${BUILD_URL}"
                fi
                '''
            }
        }
        if ("${test_mode}" == "training")
        {
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    fingerprintArtifacts: true
                )           
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && mkdir -p refer && mv inductor_log refer
                docker cp ${WORKSPACE}/refer $USER:/workspace/pytorch/
                docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report_train.py -r refer -t ${_target}"
                rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report_train.py -t ${_target}"
                '''
            }
        }
    }    

    stage('archiveArtifacts') {
        if ("${test_mode}" == "inference")
        {
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && sudo mv ${WORKSPACE}/${_target}/inductor_log/ ./ && sudo rm -rf ${_target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && sudo mv ${WORKSPACE}/${_target}/inductor_log/ ./ && sudo rm -rf ${_target}
            '''
        } 
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
    }

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if ("${test_mode}" == "inference")
        {
            if (fileExists("${WORKSPACE}/inductor_log/inductor_model_bench.html") == true){
                emailext(
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-${env._WRAPPER}-Report(Local)_${env._target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-${env._WRAPPER}-(Local)_${env._target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//inference
        if ("${test_mode}" == "training")
        {
            if (fileExists("${WORKSPACE}/inductor_log/inductor_model_training_bench.html") == true){
                emailext(
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-${env._WRAPPER}-Report(Local)_${env._target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_training_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor Training Benchmark (Local)_${env._target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }           
        }//training
    }//email
}
