env.bench_machine = "Local"
env.target = new Date().format('yyyy_MM_dd')
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE= 'ubuntu:22.04'
env.LOG_DIR = 'inductor_log'
env.NODE_LABEL = "inductor-gnr-local"

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

default_mail = 'yudong.si@intel.com'
if ('default_mail' in params) {
    echo "default_mail in params"
    if (params.default_mail != '') {
        default_mail = params.default_mail
    }
}
echo "default_mail: $default_mail"

terminate_instance = 'False'
if ('terminate_instance' in params) {
    echo "terminate_instance in params"
    if (params.terminate_instance != '') {
        terminate_instance = params.terminate_instance
    }
}
echo "terminate_instance: $terminate_instance"

backend = 'inductor'
if ('backend' in params) {
    echo "backend in params"
    if (params.backend != '') {
        backend = params.backend
    }
}
echo "backend: $backend"

extra_param = ''
if ('extra_param' in params) {
    echo "extra_param in params"
    if (params.extra_param != '') {
        extra_param = params.extra_param
    }
}
echo "extra_param: $extra_param"

test_mode = 'inference'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"

// set reference build
refer_build = ''
if( 'refer_build' in params && params.refer_build != '' ) {
    refer_build = params.refer_build
}
echo "refer_build: $refer_build"

Build_Image= 'true'
if ('Build_Image' in params) {
    echo "Build_Image in params"
    if (params.Build_Image != '') {
        Build_Image = params.Build_Image
    }
}
echo "Build_Image: $Build_Image"

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

ONEDNN_BRANCH= 'default'
if ('ONEDNN_BRANCH' in params) {
    echo "ONEDNN_BRANCH in params"
    if (params.ONEDNN_BRANCH != '') {
        ONEDNN_BRANCH = params.ONEDNN_BRANCH
    }
}
echo "ONEDNN_BRANCH: $ONEDNN_BRANCH"

DYNAMO_BENCH= "$TORCH_COMMIT"
if ('DYNAMO_BENCH' in params) {
    echo "DYNAMO_BENCH in params"
    if (params.DYNAMO_BENCH != '') {
        DYNAMO_BENCH = params.DYNAMO_BENCH
    }
}
echo "DYNAMO_BENCH: $DYNAMO_BENCH"

TORCH_AUDIO_BRANCH= 'main'
if ('TORCH_AUDIO_BRANCH' in params) {
    echo "TORCH_AUDIO_BRANCH in params"
    if (params.TORCH_AUDIO_BRANCH != '') {
        TORCH_AUDIO_BRANCH = params.TORCH_AUDIO_BRANCH
    }
}
echo "TORCH_AUDIO_BRANCH: $TORCH_AUDIO_BRANCH"

AUDIO= 'default'
if ('AUDIO' in params) {
    echo "AUDIO in params"
    if (params.AUDIO != '') {
        AUDIO = params.AUDIO
    }
}
echo "AUDIO: $AUDIO"

TORCH_TEXT_BRANCH= 'main'
if ('TORCH_TEXT_BRANCH' in params) {
    echo "TORCH_TEXT_BRANCH in params"
    if (params.TORCH_TEXT_BRANCH != '') {
        TORCH_TEXT_BRANCH = params.TORCH_TEXT_BRANCH
    }
}
echo "TORCH_TEXT_BRANCH: $TORCH_TEXT_BRANCH"

TEXT= 'default'
if ('TEXT' in params) {
    echo "TEXT in params"
    if (params.TEXT != '') {
        TEXT = params.TEXT
    }
}
echo "TEXT: $TEXT"

TORCH_VISION_BRANCH= 'main'
if ('TORCH_VISION_BRANCH' in params) {
    echo "TORCH_VISION_BRANCH in params"
    if (params.TORCH_VISION_BRANCH != '') {
        TORCH_VISION_BRANCH = params.TORCH_VISION_BRANCH
    }
}
echo "TORCH_VISION_BRANCH: $TORCH_VISION_BRANCH"

VISION= 'default'
if ('VISION' in params) {
    echo "VISION in params"
    if (params.VISION != '') {
        VISION = params.VISION
    }
}
echo "VISION: $VISION"

TORCH_DATA_BRANCH= 'main'
if ('TORCH_DATA_BRANCH' in params) {
    echo "TORCH_DATA_BRANCH in params"
    if (params.TORCH_DATA_BRANCH != '') {
        TORCH_DATA_BRANCH = params.TORCH_DATA_BRANCH
    }
}
echo "TORCH_DATA_BRANCH: $TORCH_DATA_BRANCH"

DATA= 'default'
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

TORCH_BENCH= 'default'
if ('TORCH_BENCH' in params) {
    echo "TORCH_BENCH in params"
    if (params.TORCH_BENCH != '') {
        TORCH_BENCH = params.TORCH_BENCH
    }
}
echo "TORCH_BENCH: $TORCH_BENCH"


CHANNELS= 'first'
if ('CHANNELS' in params) {
    echo "CHANNELS in params"
    if (params.CHANNELS != '') {
        CHANNELS = params.CHANNELS
    }
}
echo "CHANNELS: $CHANNELS"

HF_TOKEN= 'hf_xx'
if ('HF_TOKEN' in params) {
    echo "HF_TOKEN in params"
    if (params.HF_TOKEN != '') {
        HF_TOKEN = params.HF_TOKEN
    }
}
echo "HF_TOKEN: $HF_TOKEN"

suite= 'all'
if ('suite' in params) {
    echo "suite in params"
    if (params.suite != '') {
        suite = params.suite
    }
}
echo "suite: $suite"

report_only = 'false'
if( 'report_only' in params && params.report_only != '' ) {
    report_only = params.report_only
}
echo "report_only: $report_only"

env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._backend = "$backend"
env._extra_param = "$extra_param"
env._target = new Date().format('yyyy_MM_dd')
env._report_only = "$report_only"

env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH"
env._ONEDNN_BRANCH = "$ONEDNN_BRANCH"

env._AUDIO = "$AUDIO"
env._TEXT = "$TEXT"
env._VISION = "$VISION"
env._DATA = "$DATA"
env._TORCH_BENCH = "$TORCH_BENCH"
env._THREADS = "$THREADS"
env._CHANNELS = "$CHANNELS"
env._WRAPPER = "$WRAPPER"
env._HF_TOKEN = "$HF_TOKEN"
env._suite = "$suite"

def cleanup(){
    try {
        sh'''
            #!/usr/bin/env bash
            docker_ps=`docker ps -a -q`
            if [ -n "${docker_ps}" ];then
                docker stop ${docker_ps}
            fi
            docker container prune -f
            docker system prune -f
            docker pull ${BASE_IMAGE}
        '''
        docker.image(env.BASE_IMAGE).inside(" \
            -u root \
            -v ${WORKSPACE}:/root/workspace \
            --privileged \
        "){
        sh '''
            chmod -R 777 /root/workspace    
        '''
        }
        deleteDir()
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

def pruneOldImage(){
    sh '''
        #!/usr/bin/env bash
        old_image_id=`docker images | grep pt_inductor | awk '{print $3}'`
        old_image=`echo $old_image_id | awk '{print $1}'`
        if [ -n "${old_image}" ]; then
            docker rmi -f $old_image
        fi
        docker system prune -f
    '''
}

node(us_node){
    stage("prepare torch repo"){
        if  ("${report_only}" == "false") {
            deleteDir()
            sh'''
                #!/usr/bin/env bash
                if [ "${TORCH_COMMIT}" == "nightly" ];then
                    # clone pytorch repo
                    cd ${WORKSPACE}
                    git clone -b ${TORCH_COMMIT} ${TORCH_REPO}
                    cd pytorch
                    main_commit=`git log -n 1 --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1`
                    git checkout ${main_commit}
                else
                    # clone pytorch repo
                    cd ${WORKSPACE}
                    git clone ${TORCH_REPO}
                    cd pytorch
                    git checkout ${TORCH_COMMIT}
                fi
                commit_date=`git log -n 1 --format="%cs"`
                bref_commit=`git rev-parse --short HEAD`
                DOCKER_TAG="${commit_date}_${bref_commit}_userbm"
                echo "${DOCKER_TAG}" > ${WORKSPACE}/docker_image_tag.log
            '''
            if (fileExists("${WORKSPACE}/docker_image_tag.log")) {
                stash includes: 'docker_image_tag.log', name: 'docker_image_tag'
                archiveArtifacts  "docker_image_tag.log"
            }
        }
    }
}

node(NODE_LABEL){
    stage("prepare"){
        println('prepare......')
        // TODO: implement report_only logic
        if  ("${report_only}" == "false") {
            cleanup()
            pruneOldImage()
            retry(3){
                sleep(60)
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                    extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
            unstash 'docker_image_tag'
            sh'''
                #!/usr/bin/env bash
                mkdir -p ${WORKSPACE}/${LOG_DIR}
                mv docker_image_tag.log ${WORKSPACE}/${LOG_DIR}
            '''
        } else {
            retry(3){
                sleep(60)
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                    extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
        }
    }

    stage("trigger inductor images job"){
        if  ("${report_only}" == "false") {
            if ("${build_image}" == "true") {
                def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/${LOG_DIR}/docker_image_tag.log''').toString().trim().replaceAll("\n","")
                def image_build_job = build job: 'inductor_images', propagate: false, parameters: [             
                    [$class: 'StringParameterValue', name: 'PT_REPO', value: "${TORCH_REPO}"],
                    [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${TORCH_COMMIT}"],
                    [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
                    [$class: 'StringParameterValue', name: 'HF_TOKEN', value: "${HF_TOKEN}"],
                    [$class: 'StringParameterValue', name: 'TORCH_BENCH_COMMIT', value: "${TORCH_BENCH}"],
                ]
            }
            if (fileExists("${WORKSPACE}/${LOG_DIR}/docker_image_tag.log")) {
                archiveArtifacts  "${LOG_DIR}/docker_image_tag.log"
            }
        }
    }

    stage("prepare container"){
        if  ("${report_only}" == "false") {
            withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                sh'''
                    #!/usr/bin/env bash
                    docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
                    docker pull ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
                '''
            }
        }
    }

    stage("benchmark") {
        if  ("${report_only}" == "false")
        {
            sh '''
            #!/usr/bin/env bash
            docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
            docker run -tid --name $USER \
                --privileged \
                --env https_proxy=${https_proxy} \
                --env http_proxy=${http_proxy} \
                --env HF_HUB_TOKEN=$HF_TOKEN \
                --net host --shm-size 20G \
                -v ~/.cache:/root/.cache \
                -v ${WORKSPACE}/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} \
                ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
            
            docker cp scripts/userbenchmark/version_collect_userbm.sh $USER:/workspace/pytorch
            docker cp scripts/userbenchmark/cpu_usebm.sh $USER:/workspace/pytorch
            docker cp scripts/userbenchmark/cpu_usebm_train.sh $USER:/workspace/pytorch
            docker exec -i $USER bash -c "bash version_collect_userbm.sh ${LOG_DIR} $DYNAMO_BENCH"

            if [ $test_mode == "user_benchmark_infer" ]; then
                docker exec -i $USER bash -c "bash cpu_usebm.sh"
            elif [ $test_mode == "user_benchmark_train" ]; then
                docker exec -i $USER bash -c "bash cpu_usebm_train.sh"
            elif [ $test_mode == "user_benchmark" ]; then
                docker exec -i $USER bash -c "bash cpu_usebm.sh"
            fi

            docker exec -i $USER bash -c "chmod 777 -R /workspace/pytorch/${LOG_DIR}"
            '''
        }
    }
    
    
    // Add raw log artifact stage in advance to avoid crash in report generate stage
    stage("archive raw test results"){
        if  ("${report_only}" == "false") {
            sh '''
            #!/usr/bin/env bash
            if [ -d ${WORKSPACE}/raw_log ];then
                rm -rf ${WORKSPACE}/raw_log
            fi
            if [ -d ${WORKSPACE}/${target} ];then
                rm -rf ${WORKSPACE}/${target}
            fi
            cp -r ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/raw_log
            mkdir ${WORKSPACE}/${target}
            mv ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/${target}/
        '''
        }
        archiveArtifacts artifacts: "**/raw_log/**", fingerprint: true
    }
    
    stage("stop docker") {
        sh'''
            #!/usr/bin/env bash
            docker_ps=`docker ps -a -q`
            if [ -n "${docker_ps}" ];then
                docker stop ${docker_ps}
            fi
        '''
    }

    stage("generate report"){
        if(refer_build != '0') {
            copyArtifacts(
                projectName: currentBuild.projectName,
                selector: specific("${refer_build}"),
                fingerprintArtifacts: true,
                target: "refer",
            )           
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} 
            cp scripts/userbenchmark/report_userbm.py ${WORKSPACE}
            python report_userbm.py -r refer -t ${_target} --url ${BUILD_URL}
            rm -rf refer 
            '''
        }
    }    

    stage('archiveArtifacts') {
        // Remove raw log fistly in case inducto_log will be artifact more than 2 times
        sh '''
        #!/usr/bin/env bash
        rm -rf ${WORKSPACE}/raw_log
        '''
        if ("${test_mode}" == "user_benchmark")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard
            cp -r  ${WORKSPACE}/${_target} $HOME/inductor_dashboard
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard/Train
            cp -r  ${WORKSPACE}/${_target} $HOME/inductor_dashboard/Train
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
            '''
        } 
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
    }

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="${default_mail}"
        }
        if (fileExists("${WORKSPACE}/inductor_log/userbenchmark_model_bench.html") == true){
            emailext(
                subject: "Userbenchmark-Regular-Report(AWS)_${env._target}",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="inductor_log/userbenchmark_model_bench.html"}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Userbenchmark-Regular-Report(AWS)_${env._target}",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}
