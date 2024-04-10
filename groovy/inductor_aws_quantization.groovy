NODE_LABEL = 'yudongsi-mlt-ace'
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

instance_ids = '0'
if ('instance_ids' in params) {
    echo "instance_ids in params"
    if (params.instance_ids != '') {
        instance_ids = params.instance_ids
    }
}
echo "instance_ids: $instance_ids"

instance_name = 'icx-ondemand'
if ('instance_name' in params) {
    echo "instance_name in params"
    if (params.instance_name != '') {
        instance_name = params.instance_name
    }
}
echo "instance_name: $instance_name"

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

precision = 'float32'
if ('precision' in params) {
    echo "precision in params"
    if (params.precision != '') {
        precision = params.precision
    }
}
echo "precision: $precision"

test_mode = 'performance'
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

Build_Image= 'true'
if ('Build_Image' in params) {
    echo "Build_Image in params"
    if (params.Build_Image != '') {
        Build_Image = params.Build_Image
    }
}
echo "Build_Image: $Build_Image"

IMAGE_BUILD_NODE= 'inductor_image'
if ('IMAGE_BUILD_NODE' in params) {
    echo "IMAGE_BUILD_NODE in params"
    if (params.IMAGE_BUILD_NODE != '') {
        IMAGE_BUILD_NODE = params.IMAGE_BUILD_NODE
    }
}
echo "IMAGE_BUILD_NODE: $IMAGE_BUILD_NODE"

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

env._terminate_ins = "$terminate_instance"
env._instance_id = "$instance_ids"
env._instance_name = "$instance_name"
env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._backend = "$backend"
env._extra_param = "$extra_param"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._gh_token = "$gh_token"
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

node(NODE_LABEL){
    stage("Find or create instance"){
        deleteDir()
        checkout scm
        if  ("${report_only}" == "false")
        {
            sh'''
            #!/usr/bin/env bash
            cd ${WORKSPACE}/scripts/aws/
            while true;
            do
                bash find_instance.sh ${_instance_name} ${_instance_id} 2>&1 | tee ${WORKSPACE}/instance_id.txt
                ins_id=`cat ${WORKSPACE}/instance_id.txt`
                if [ $ins_id != "waiting_instance" ]; then
                    echo "ins_id : $ins_id"
                    break
                else
                    echo "Waiting for avaliable instance, will check after 10 min..."
                    sleep 10m
                fi
            done
            '''
        } else {
            echo "report_only mode, will directly use the instance_id"
            sh'''
            #!/usr/bin/env bash
            echo ${_instance_id} > ${WORKSPACE}/instance_id.txt
            echo "ins_id : ${_instance_id}"
            '''
        }
    }
    stage("start instance")
    {
        sh '''
        #!/usr/bin/env bash
        ins_id=`cat ${WORKSPACE}/instance_id.txt`
        cd $HOME && $aws ec2 start-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m
        init_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        echo init_ip is $init_ip
        ssh -o StrictHostKeyChecking=no ubuntu@${init_ip} "pwd"
        '''
    }
    stage("prepare scripts & benchmark") {
        if  ("${report_only}" == "false")
        {
            sh '''
            #!/usr/bin/env bash
            ins_id=`cat ${WORKSPACE}/instance_id.txt`
            current_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            ssh ubuntu@${current_ip} "if [ ! -d /home/ubuntu/docker ]; then mkdir -p /home/ubuntu/docker; fi"
            scp ${WORKSPACE}/scripts/aws/inductor_weights.sh ubuntu@${current_ip}:/home/ubuntu
            scp ${WORKSPACE}/scripts/aws/docker_prepare.sh ubuntu@${current_ip}:/home/ubuntu
            ssh ubuntu@${current_ip} "bash docker_prepare.sh"
            scp ${WORKSPACE}/scripts/modelbench/pkill.sh ubuntu@${current_ip}:/home/ubuntu
            scp ${WORKSPACE}/scripts/modelbench/entrance.sh ubuntu@${current_ip}:/home/ubuntu
            scp ${WORKSPACE}/docker/Dockerfile ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/launch.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/version_collect.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_test.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_train.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_quant_performance.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/hf_quant_test.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/numa_launcher.py ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_dynamic_quant.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_quant_accuracy.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_quant_acc.py ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_quant_acc_fp32.py ubuntu@${current_ip}:/home/ubuntu/docker

            ssh ubuntu@${current_ip} "bash pkill.sh"
            ssh ubuntu@${current_ip} "nohup bash entrance.sh ${_target} ${_precision} ${_test_mode} ${_shape} ${_TORCH_REPO} ${_TORCH_BRANCH} ${_TORCH_COMMIT} ${_DYNAMO_BENCH} ${_AUDIO} ${_TEXT} ${_VISION} ${_DATA} ${_TORCH_BENCH} ${_THREADS} ${_CHANNELS} ${_WRAPPER} ${_HF_TOKEN} ${_backend} torchbench resnet50 ${_TORCH_COMMIT} ${_TORCH_COMMIT} accuracy crash ${_extra_param} &>/dev/null &" &
            '''
        }
    }
    stage("trigger inductor images job"){
            if ("${Build_Image}" == "true") {
                def image_build_job = build job: 'inductor_images', propagate: false, parameters: [
                    [$class: 'StringParameterValue', name: 'NODE_LABEL', value: "${IMAGE_BUILD_NODE}"],
                    [$class: 'StringParameterValue', name: 'BASE_IMAGE', value: "${BASE_IMAGE}"],                
                    [$class: 'StringParameterValue', name: 'PT_REPO', value: "${TORCH_REPO}"],
                    [$class: 'StringParameterValue', name: 'PT_BRANCH', value: "${TORCH_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${TORCH_COMMIT}"],
                    [$class: 'StringParameterValue', name: 'TORCH_VISION_BRANCH', value: "${TORCH_VISION_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'TORCH_VISION_COMMIT', value: "${VISION}"],
                    [$class: 'StringParameterValue', name: 'TORCH_TEXT_BRANCH', value: "${TORCH_TEXT_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'TORCH_TEXT_COMMIT', value: "${TEXT}"],
                    [$class: 'StringParameterValue', name: 'TORCH_DATA_BRANCH', value: "${TORCH_DATA_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'TORCH_DATA_COMMIT', value: "${DATA}"],
                    [$class: 'StringParameterValue', name: 'TORCH_AUDIO_BRANCH', value: "${TORCH_AUDIO_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'TORCH_AUDIO_COMMIT', value: "${AUDIO}"],
                    [$class: 'StringParameterValue', name: 'TORCH_BENCH_BRANCH', value: "${TORCH_BENCH_BRANCH}"],
                    [$class: 'StringParameterValue', name: 'TORCH_BENCH_COMMIT', value: "${TORCH_BENCH}"],
                    [$class: 'StringParameterValue', name: 'BENCH_COMMIT', value: "${DYNAMO_BENCH}"],
                    [$class: 'StringParameterValue', name: 'tag', value: "${env._target}_aws"],
                    [$class: 'StringParameterValue', name: 'HF_TOKEN', value: "${HF_TOKEN}"],
                ]
            }
    }
    stage("log query") {
        sh '''
        #!/usr/bin/env bash
        set +e
        if [ "${_WRAPPER}" == "cpp" ]; then
            reboot_time=48
            echo "cppwrapper"
        else
            reboot_time=44
        fi
        ins_id=`cat ${WORKSPACE}/instance_id.txt`        
        for t in {1..100}
        do
            current_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            timeout 2m ssh ubuntu@${current_ip} "test -f /home/ubuntu/docker/finished_${_precision}_${_test_mode}_${_shape}.txt"
            if [ $? -eq 0 ]; then
                if [ -d ${WORKSPACE}/${_target} ]; then
                    rm -rf ${WORKSPACE}/${_target}
                fi
                mkdir -p ${WORKSPACE}/${_target}
                scp -r ubuntu@${current_ip}:/home/ubuntu/docker/inductor_log ${WORKSPACE}/${_target}
                break
            else
                sleep 1h
                echo $t
                if [ $t -eq $reboot_time ]; then
                    echo restart instance now...
                    $aws ec2 stop-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m && $aws ec2 start-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m && current_ip=$($aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text) && echo update_ip $current_ip || echo $current_ip
                    ssh -o StrictHostKeyChecking=no ubuntu@${current_ip} "pwd"
                    scp -r ubuntu@${current_ip}:/home/ubuntu/docker/inductor_log ${WORKSPACE}/${_target}
                    break
                fi
            fi
        done
        '''
    }
    // Add raw log artifact stage in advance to avoid crash in report generate stage
    stage("archive raw test results"){
        sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard
            cp -r  ${WORKSPACE}/${_target} ${WORKSPACE}/raw_log
        '''
        archiveArtifacts artifacts: "**/raw_log/**", fingerprint: true
    }
    stage("stop or terminate instance")
    {
        try{
            sh '''
            #!/usr/bin/env bash
            ins_id=`cat ${WORKSPACE}/instance_id.txt`
            $aws ec2 stop-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m
            if [ "$_terminate_ins" == "True" ]; then
                $aws ec2 terminate-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m
            fi
            '''
        }catch(err){
            echo err.getMessage()   
        }
    }
    stage("generate report"){
        if(refer_build != '0') {
            copyArtifacts(
                projectName: currentBuild.projectName,
                selector: specific("${refer_build}"),
                fingerprintArtifacts: true
            )           
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && rm inductor_log/*.html && rm inductor_log/*.xlsx
            mkdir -p refer && cp -r inductor_log refer && rm -rf inductor_log
            cp scripts/modelbench/report_quant_perf.py ${WORKSPACE} && python report_quant_perf.py -r refer -t ${_target} --url ${BUILD_URL}      
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
        if (fileExists("${WORKSPACE}/inductor_log/quantization_model_bench.html") == true){
            emailext(
                subject: "Quantization-Regular-Report(AWS)_${env._target}",
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
