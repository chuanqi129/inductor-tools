env.bench_machine = "Local"
env.target = new Date().format('yyyy_MM_dd')
env.DOCKER_IMAGE_NAMESPACE = 'gar-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE= 'ubuntu:22.04'
env.LOG_DIR = 'inductor_log'
env.NODE_LABEL = params.NODE_LABEL

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

extra_param = ''
if ('extra_param' in params) {
    echo "extra_param in params"
    if (params.extra_param != '') {
        extra_param = params.extra_param
    }
}
echo "extra_param: $extra_param"

test_mode = 'performance'
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

BASE_IMAGE= 'ubuntu:20.04'
if ('BASE_IMAGE' in params) {
    echo "BASE_IMAGE in params"
    if (params.BASE_IMAGE != '') {
        BASE_IMAGE = params.BASE_IMAGE
    }
}
echo "BASE_IMAGE: $BASE_IMAGE"

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

HF_TOKEN= 'hf_xx'
if ('HF_TOKEN' in params) {
    echo "HF_TOKEN in params"
    if (params.HF_TOKEN != '') {
        HF_TOKEN = params.HF_TOKEN
    }
}
echo "HF_TOKEN: $HF_TOKEN"


dash_board = 'false'
if( 'dash_board' in params && params.dash_board != '' ) {
    dash_board = params.dash_board
}
echo "dash_board: $dash_board"

report_only = 'false'
if( 'report_only' in params && params.report_only != '' ) {
    report_only = params.report_only
}
echo "report_only: $report_only"

dashboard_title = 'default'
if( 'dashboard_title' in params && params.dashboard_title != '' ) {
    dashboard_title = params.dashboard_title
}
echo "dashboard_title: $dashboard_title"

QUANTIZATION_PATH = 'all'
if( 'QUANTIZATION_PATH' in params && params.QUANTIZATION_PATH != '' ) {
    QUANTIZATION_PATH = params.QUANTIZATION_PATH
}
echo "QUANTIZATION_PATH: $QUANTIZATION_PATH"

MODELS = 'resnet50'
if( 'MODELS' in params && params.MODELS != '' ) {
    MODELS = params.MODELS
}
echo "MODELS: $MODELS"

env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._extra_param = "$extra_param"
env._target = new Date().format('yyyy_MM_dd')
env._dash_board = "$dash_board"
env._report_only = "$report_only"
env._dashboard_title = "$dashboard_title"

env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH"
env._QUANTIZATION_PATH = "$QUANTIZATION_PATH"
env._MODELS = "$MODELS"

env._AUDIO = "$AUDIO"
env._TEXT = "$TEXT"
env._VISION = "$VISION"
env._DATA = "$DATA"
env._TORCH_BENCH = "$TORCH_BENCH"
env._THREADS = "$THREADS"
env._CHANNELS = "$CHANNELS"
env._WRAPPER = "$WRAPPER"
env._HF_TOKEN = "$HF_TOKEN"

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
                DOCKER_TAG="${commit_date}_${bref_commit}_quant"
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
                def image_build_job = build job: 'inductor_images_local_py310', propagate: false, parameters: [             
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
        if  ("${report_only}" == "false") {
            sh '''
                #!/usr/bin/env bash
                docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
                docker run -tid --name inductor_quant \
                    --privileged \
                    --env https_proxy=${https_proxy} \
                    --env http_proxy=${http_proxy} \
                    --env HF_HUB_TOKEN=$HF_TOKEN \
                    --net host --shm-size 20G \
                    -v ~/.cache:/root/.cache \
                    -v ${WORKSPACE}/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} \
                    ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
                docker cp scripts/modelbench/quant/version_collect_quant.sh inductor_quant:/workspace/pytorch
                docker cp scripts/modelbench/quant/inductor_quant_performance.sh inductor_quant:/workspace/pytorch
                docker cp scripts/modelbench/quant/inductor_quant_accuracy.sh inductor_quant:/workspace/pytorch
                docker cp scripts/modelbench/quant/inductor_quant_acc.py inductor_quant:/workspace/benchmark
                docker cp scripts/modelbench/quant/hf_quant_test.sh inductor_quant:/workspace/pytorch
                docker cp scripts/modelbench/quant/inductor_dynamic_quant.sh inductor_quant:/workspace/pytorch
                docker cp scripts/modelbench/quant/numa_launcher.py inductor_quant:/workspace/pytorch
                docker exec -i inductor_quant bash -c "bash version_collect_quant.sh ${LOG_DIR} $DYNAMO_BENCH"

                prepare_imagenet(){
                    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
                    mkdir -p ${WORKSPACE}/imagenet/val && mv ILSVRC2012_img_val.tar ${WORKSPACE}/imagenet/val && cd ${WORKSPACE}/imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
                    wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
                    bash valprep.sh
                }
                if [ $test_mode == "performance" ]; then
                    docker exec -i inductor_quant bash -c "bash inductor_quant_performance.sh ${LOG_DIR}"
                elif [ $test_mode == "accuracy" ]; then
                    if [ ! -d "${WORKSPACE}/imagenet" ];then
                        prepare_imagenet
                    fi
                    docker cp ${WORKSPACE}/imagenet inductor_quant:/workspace/benchmark/
                    docker exec -i inductor_quant bash -c "bash inductor_quant_accuracy.sh ${LOG_DIR}"
                elif [ $test_mode == "all" ]; then
                    if [ ! -d "${WORKSPACE}/imagenet" ];then
                        prepare_imagenet
                    fi
                    docker exec -i inductor_quant bash -c "bash inductor_quant_performance.sh ${LOG_DIR}"
                    docker cp ${WORKSPACE}/imagenet inductor_quant:/workspace/benchmark/
                    docker exec -i inductor_quant bash -c "bash inductor_quant_accuracy.sh ${LOG_DIR}"
                    docker exec -i inductor_quant bash -c "bash inductor_dynamic_quant.sh ${LOG_DIR}"
                fi
                
                docker exec -i inductor_quant bash -c "chmod 777 -R /workspace/pytorch/${LOG_DIR}"
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
            cp scripts/modelbench/quant/report_quant_perf.py ${WORKSPACE}
            source /root/miniforge3/bin/activate base
            python3 report_quant_perf.py -r refer -t ${target} --url ${BUILD_URL}
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
        if ("${test_mode}" == "all")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard
            cp -r  ${WORKSPACE}/${target} $HOME/inductor_dashboard
            cd ${WORKSPACE} && mv ${WORKSPACE}/${target}/inductor_log/ ./ && rm -rf ${target}
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
        if (fileExists("${WORKSPACE}/inductor_log/quantization_model_bench.html") == true){
            emailext(
                subject: "Quantization-Regular-Report(AWS)_${env.target}",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="inductor_log/quantization_model_bench.html"}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Quantization-Regular-Report(AWS)_${env._target}",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}
