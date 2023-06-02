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

instance_ids = 'i-03aa90bc2017ba908'
if ('instance_ids' in params) {
    echo "instance_ids in params"
    if (params.instance_ids != '') {
        instance_ids = params.instance_ids
    }
}
echo "instance_ids: $instance_ids"

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

dash_board = 'false'
if( 'dash_board' in params && params.dash_board != '' ) {
    dash_board = params.dash_board
}
echo "dash_board: $dash_board"

env._aws_id = "$instance_ids"
env._reference = "$refer_build"
env._test_mode = "$test_mode"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._gh_token = "$gh_token"
env._dash_board = "$dash_board"

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

node(NODE_LABEL){
    stage("start instance")
    {
        deleteDir()
        checkout scm
        sh '''
        #!/usr/bin/env bash
        cd $HOME && $aws ec2 start-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
        init_ip=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        echo init_ip is $init_ip
        ssh -o StrictHostKeyChecking=no ubuntu@${init_ip} "pwd"
        '''
    }
    stage("prepare scripts & benchmark") {
        retry(3){
            sh '''
            #!/usr/bin/env bash
            current_ip=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            ssh ubuntu@${current_ip} "if [ ! -d /home/ubuntu/docker ]; then mkdir -p /home/ubuntu/docker; fi"
            scp ${WORKSPACE}/scripts/modelbench/entrance.sh ubuntu@${current_ip}:/home/ubuntu
            scp ${WORKSPACE}/docker/Dockerfile ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/launch.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_test.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_train.sh ubuntu@${current_ip}:/home/ubuntu/docker
            ssh ubuntu@${current_ip} "nohup bash entrance.sh ${_target} ${_precision} ${_test_mode} ${_shape} ${_TORCH_REPO} ${_TORCH_BRANCH} ${_TORCH_COMMIT} ${_DYNAMO_BENCH} ${_AUDIO} ${_TEXT} ${_VISION} ${_DATA} ${_TORCH_BENCH} ${_THREADS} ${_CHANNELS} &>/dev/null &" &
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
                    [$class: 'StringParameterValue', name: 'tag', value: "${env._target}"],
                ]
            }
    }
    stage("log query") {
        sh '''
        #!/usr/bin/env bash
        set +e
        current_ip=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        for t in {1..25}
        do
            ssh ubuntu@${current_ip} "test -f /home/ubuntu/docker/finished_${_precision}_${_test_mode}_${_shape}.txt"
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
                if [ $t -eq 24 ]; then
                    echo restart instance now...
                    $aws ec2 stop-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m && $aws ec2 start-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m && current_ip=$($aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text) && echo update_ip $current_ip || echo $current_ip
                    ssh -o StrictHostKeyChecking=no ubuntu@${current_ip} "pwd"
                fi
            fi
        done
        '''
    }
    stage("stop instance")
    {
        try{
            sh '''
            #!/usr/bin/env bash
            $aws ec2 stop-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
            '''
        }catch(err){
            echo err.getMessage()   
        }
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
                cd ${WORKSPACE} && mkdir -p refer && cp -r inductor_log refer && rm -rf inductor_log
                if [ ${_dash_board} == "true" ]; then
                     cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target} -m ${_THREADS} --gh_token ${_gh_token} && rm -rf refer
                else
                     cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target} -m ${_THREADS} --md_off --precision ${_precision} && rm -rf refer
                fi
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -t ${_target} -m all --md_off --precision ${_precision}
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
                cd ${WORKSPACE} && mkdir -p refer && cp -r inductor_log refer && rm -rf inductor_log
                cp scripts/modelbench/report_train.py ${WORKSPACE} && python report_train.py -r refer -t ${_target} && rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && cp scripts/modelbench/report_train.py ${WORKSPACE} && python report_train.py -t ${_target}
                '''
            }
        }
    }    

    stage('archiveArtifacts') {
        if ("${test_mode}" == "inference")
        {
            sh '''
            #!/usr/bin/env bash
            cp -r  ${WORKSPACE}/${_target} $HOME/inductor_dashboard
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
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
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if ("${test_mode}" == "inference")
        {
            if (fileExists("${WORKSPACE}/inductor_log/inductor_model_bench.html") == true){
                emailext(
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-Report(AWS)_${env._target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-(AWS)_${env._target}",
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
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-Report(AWS)_${env._target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_training_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor Training Benchmark (AWS)_${env._target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }           
        }//training
    }//email
}
