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

image_tag = 'nightly'
if ('image_tag' in params) {
    echo "image_tag in params"
    if (params.image_tag != '') {
        image_tag = params.image_tag
    }
}
echo "image_tag: $image_tag"

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

env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._gh_token = "$gh_token"
env._dash_board = "$dash_board"
env._dashboard_title = "$dashboard_title"
env._THREADS = "$THREADS"
env._CHANNELS = "$CHANNELS"
env._WRAPPER = "$WRAPPER"
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env._image_tag = "$image_tag"

env.INDUCTOR_CACHE = '/home2/yudongsi/.cache'

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

    stage("benchmark") {
        sh '''
        #!/usr/bin/env bash    
        docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 1G -v ${INDUCTOR_CACHE}:/root/.cache -v ${WORKSPACE}/${_target}/inductor_log:/workspace/pytorch/${_target} ${DOCKER_IMAGE_NAMESPACE}:${_image_tag}
        docker cp scripts/modelbench/inductor_test.sh $USER:/workspace/pytorch
        docker cp scripts/modelbench/inductor_train.sh $USER:/workspace/pytorch
        docker cp scripts/modelbench/report.py $USER:/workspace/pytorch
        docker exec -i $USER bash -c "export TRANSFORMERS_OFFLINE=1;bash inductor_test.sh ${_THREADS} ${_CHANNELS} ${_precision} ${_shape} inductor_log nightly ${_WRAPPER}"
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
                     docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -r refer -t ${_target} -m ${_THREADS} --gh_token ${_gh_token} --dashboard ${_dashboard_title}"
                else
                     docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -r refer -t ${_target} -m ${_THREADS} --md_off --precision ${_precision}"
                fi
                rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                if [ ${_dash_board} == "true" ]; then
                    docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -t ${_target} -m ${_THREADS} --gh_token ${_gh_token} --dashboard ${_dashboard_title} --precision ${_precision}"
                else
                     docker exec -i $USER bash -c "pip install pandas styleframe PyGithub beautifulsoup4;python report.py -t ${_target} -m ${_THREADS} --md_off --precision ${_precision}"
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
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
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
