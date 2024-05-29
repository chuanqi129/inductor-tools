NODE_LABEL = 'yudongsi-mlt-ace'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

debug = 'True'
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

debug_mail = 'chuanqi.wang@intel.com'
if ('debug_mail' in params) {
    echo "debug_mail in params"
    if (params.debug_mail != '') {
        debug_mail = params.debug_mail
    }
}
echo "debug_mail: $debug_mail"

default_mail = 'chuanqi.wang@intel.com'
if ('default_mail' in params) {
    echo "default_mail in params"
    if (params.default_mail != '') {
        default_mail = params.default_mail
    }
}
echo "default_mail: $default_mail"

instance_name = 'icx-guilty-search'
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

gh_token = ''
if( 'gh_token' in params && params.gh_token != '' ) {
    gh_token = params.gh_token
}
echo "gh_token: $gh_token"

BASE_IMAGE= 'ubuntu:22.04'
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

TORCH_BRANCH= 'main'
if ('TORCH_BRANCH' in params) {
    echo "TORCH_BRANCH in params"
    if (params.TORCH_BRANCH != '') {
        TORCH_BRANCH = params.TORCH_BRANCH
    }
}
echo "TORCH_BRANCH: $TORCH_BRANCH"

TORCH_START_COMMIT= "$TORCH_START_COMMIT"
if ('TORCH_START_COMMIT' in params) {
    echo "TORCH_START_COMMIT in params"
    if (params.TORCH_START_COMMIT != '') {
        TORCH_START_COMMIT = params.TORCH_START_COMMIT
    }
}
echo "TORCH_START_COMMIT: $TORCH_START_COMMIT"

TORCH_END_COMMIT= "$TORCH_START_COMMIT"
if ('TORCH_END_COMMIT' in params) {
    echo "TORCH_END_COMMIT in params"
    if (params.TORCH_END_COMMIT != '') {
        TORCH_END_COMMIT = params.TORCH_END_COMMIT
    }
}
echo "TORCH_END_COMMIT: $TORCH_END_COMMIT"

TORCH_COMMIT= "$TORCH_END_COMMIT"
if ('TORCH_COMMIT' in params) {
    echo "TORCH_COMMIT in params"
    if (params.TORCH_COMMIT != '') {
        TORCH_COMMIT = params.TORCH_COMMIT
    }
}
echo "TORCH_COMMIT: $TORCH_COMMIT"

DYNAMO_BENCH_COMMIT= "$TORCH_COMMIT"
if ('DYNAMO_BENCH_COMMIT' in params) {
    echo "DYNAMO_BENCH_COMMIT in params"
    if (params.DYNAMO_BENCH_COMMIT != '') {
        DYNAMO_BENCH_COMMIT = params.DYNAMO_BENCH_COMMIT
    }
}
echo "DYNAMO_BENCH_COMMIT: $DYNAMO_BENCH_COMMIT"

TORCH_AUDIO_BRANCH= 'main'
if ('TORCH_AUDIO_BRANCH' in params) {
    echo "TORCH_AUDIO_BRANCH in params"
    if (params.TORCH_AUDIO_BRANCH != '') {
        TORCH_AUDIO_BRANCH = params.TORCH_AUDIO_BRANCH
    }
}
echo "TORCH_AUDIO_BRANCH: $TORCH_AUDIO_BRANCH"

TORCH_AUDIO_COMMIT= 'default'
if ('TORCH_AUDIO_COMMIT' in params) {
    echo "TORCH_AUDIO_COMMIT in params"
    if (params.TORCH_AUDIO_COMMIT != '') {
        TORCH_AUDIO_COMMIT = params.TORCH_AUDIO_COMMIT
    }
}
echo "TORCH_AUDIO_COMMIT: $TORCH_AUDIO_COMMIT"

TORCH_TEXT_BRANCH= 'main'
if ('TORCH_TEXT_BRANCH' in params) {
    echo "TORCH_TEXT_BRANCH in params"
    if (params.TORCH_TEXT_BRANCH != '') {
        TORCH_TEXT_BRANCH = params.TORCH_TEXT_BRANCH
    }
}
echo "TORCH_TEXT_BRANCH: $TORCH_TEXT_BRANCH"

TORCH_TEXT_COMMIT= 'default'
if ('TORCH_TEXT_COMMIT' in params) {
    echo "TORCH_TEXT_COMMIT in params"
    if (params.TORCH_TEXT_COMMIT != '') {
        TORCH_TEXT_COMMIT = params.TORCH_TEXT_COMMIT
    }
}
echo "TORCH_TEXT_COMMIT: $TORCH_TEXT_COMMIT"

TORCH_VISION_BRANCH= 'main'
if ('TORCH_VISION_BRANCH' in params) {
    echo "TORCH_VISION_BRANCH in params"
    if (params.TORCH_VISION_BRANCH != '') {
        TORCH_VISION_BRANCH = params.TORCH_VISION_BRANCH
    }
}
echo "TORCH_VISION_BRANCH: $TORCH_VISION_BRANCH"

TORCH_VISION_COMMIT= 'default'
if ('TORCH_VISION_COMMIT' in params) {
    echo "TORCH_VISION_COMMIT in params"
    if (params.TORCH_VISION_COMMIT != '') {
        TORCH_VISION_COMMIT = params.TORCH_VISION_COMMIT
    }
}
echo "TORCH_VISION_COMMIT: $TORCH_VISION_COMMIT"

TORCH_DATA_BRANCH= 'main'
if ('TORCH_DATA_BRANCH' in params) {
    echo "TORCH_DATA_BRANCH in params"
    if (params.TORCH_DATA_BRANCH != '') {
        TORCH_DATA_BRANCH = params.TORCH_DATA_BRANCH
    }
}
echo "TORCH_DATA_BRANCH: $TORCH_DATA_BRANCH"

TORCH_DATA_COMMIT= 'default'
if ('TORCH_DATA_COMMIT' in params) {
    echo "TORCH_DATA_COMMIT in params"
    if (params.TORCH_DATA_COMMIT != '') {
        TORCH_DATA_COMMIT = params.TORCH_DATA_COMMIT
    }
}
echo "TORCH_DATA_COMMIT: $TORCH_DATA_COMMIT"

TORCH_BENCH_BRANCH= 'main'
if ('TORCH_BENCH_BRANCH' in params) {
    echo "TORCH_BENCH_BRANCH in params"
    if (params.TORCH_BENCH_BRANCH != '') {
        TORCH_BENCH_BRANCH = params.TORCH_BENCH_BRANCH
    }
}
echo "TORCH_BENCH_BRANCH: $TORCH_BENCH_BRANCH"

TORCH_BENCH_COMMIT= 'default'
if ('TORCH_BENCH_COMMIT' in params) {
    echo "TORCH_BENCH_COMMIT in params"
    if (params.TORCH_BENCH_COMMIT != '') {
        TORCH_BENCH_COMMIT = params.TORCH_BENCH_COMMIT
    }
}
echo "TORCH_BENCH_COMMIT: $TORCH_BENCH_COMMIT"

THREADS= 'multiple'
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

scenario= 'accuracy'
if ('scenario' in params) {
    echo "scenario in params"
    if (params.scenario != '') {
        scenario = params.scenario
    }
}
echo "scenario: $scenario"

perf_ratio= '1.1'
if ('perf_ratio' in params) {
    echo "perf_ratio in params"
    if (params.perf_ratio != '') {
        perf_ratio = params.perf_ratio
    }
}
echo "perf_ratio: $perf_ratio"

kind= 'crash'
if ('kind' in params) {
    echo "kind in params"
    if (params.kind != '') {
        kind = params.kind
    }
}
echo "kind: $kind"

suite= 'torchbench'
if ('suite' in params) {
    echo "suite in params"
    if (params.suite != '') {
        suite = params.suite
    }
}
echo "suite: $suite"

model= 'resnet50'
if ('model' in params) {
    echo "model in params"
    if (params.model != '') {
        model = params.model
    }
}
echo "model: $model"

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

env._instance_name = "$instance_name"
env._test_mode = "$test_mode"
env._backend = "$backend"
env._extra_param = "$extra_param"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._gh_token = "$gh_token"
env._suite = "$suite"
env._model = "$model"
env._scenario = "$scenario"
env._kind = "$kind"
env._perf_ratio = "$perf_ratio"

env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._start_commit = "$TORCH_START_COMMIT"
env._end_commit = "$TORCH_END_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH_COMMIT"

env._AUDIO = "$TORCH_AUDIO_COMMIT"
env._TEXT = "$TORCH_TEXT_COMMIT"
env._VISION = "$TORCH_VISION_COMMIT"
env._DATA = "$TORCH_DATA_COMMIT"
env._TORCH_BENCH = "$TORCH_BENCH_COMMIT"
env._THREADS = "$THREADS"
env._CHANNELS = "$CHANNELS"
env._WRAPPER = "$WRAPPER"
env._HF_TOKEN = "$HF_TOKEN"

node(NODE_LABEL){
    stage("Find or create instance"){
        deleteDir()
        checkout scm
        sh'''
        #!/usr/bin/env bash
        cd ${WORKSPACE}/scripts/aws/
        while true;
        do
            bash find_instance.sh ${instance_name} 2>&1 | tee ${WORKSPACE}/instance_id.txt
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
        sh '''
        #!/usr/bin/env bash
        ins_id=`cat ${WORKSPACE}/instance_id.txt`
        current_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        ssh ubuntu@${current_ip} "if [ ! -d /home/ubuntu/docker ]; then mkdir -p /home/ubuntu/docker; fi"
        scp ${WORKSPACE}/scripts/aws/docker_prepare.sh ubuntu@${current_ip}:/home/ubuntu
        ssh ubuntu@${current_ip} "bash docker_prepare.sh"
        
        scp ${WORKSPACE}/scripts/modelbench/pkill.sh ubuntu@${current_ip}:/home/ubuntu
        scp ${WORKSPACE}/scripts/modelbench/entrance_quant.sh ubuntu@${current_ip}:/home/ubuntu
        scp ${WORKSPACE}/scripts/modelbench/launch_quant.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/docker/Dockerfile ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/bisect_search_quant.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/bisect_run_test_quant.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/inductor_single_run.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/quant_single_run.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/hf_quant_test.sh ubuntu@${current_ip}:/home/ubuntu/docker
        scp ${WORKSPACE}/scripts/modelbench/inductor_quant_acc.py ubuntu@${current_ip}:/home/ubuntu/docker
        ssh ubuntu@${current_ip} "bash pkill.sh"
        ssh ubuntu@${current_ip} "nohup bash entrance_quant.sh ${_target} ${_precision} ${_test_mode} ${_shape} ${_TORCH_REPO} ${_TORCH_BRANCH} ${_TORCH_COMMIT} ${_DYNAMO_BENCH} ${_AUDIO} ${_TEXT} ${_VISION} ${_DATA} ${_TORCH_BENCH} ${_THREADS} ${_CHANNELS} ${_WRAPPER} ${_HF_TOKEN} ${_backend} ${_suite} ${_model} ${_start_commit} ${_end_commit} ${_scenario} ${_kind} ${_perf_ratio} ${_extra_param} &>/dev/null &" &
        '''
    }
    stage("log query") {
        sh '''
        #!/usr/bin/env bash
        set +e
        reboot_time=60
        ins_id=`cat ${WORKSPACE}/instance_id.txt`     
        for t in {1..70}
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
                sleep 10m
                echo $t
                if [ $t -eq $reboot_time ]; then
                    echo restart instance now...
                    $aws ec2 stop-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m && $aws ec2 start-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m && current_ip=$($aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text) && echo update_ip $current_ip || echo $current_ip
                    ssh -o StrictHostKeyChecking=no ubuntu@${current_ip} "pwd"
                    if [ -d ${WORKSPACE}/${_target} ]; then
                        rm -rf ${WORKSPACE}/${_target}
                    fi
                    mkdir -p ${WORKSPACE}/${_target}
                    scp -r ubuntu@${current_ip}:/home/ubuntu/docker/inductor_log ${WORKSPACE}/${_target}
                    break
                fi
            fi
        done
        '''
    }
    stage("terminate instance")
    {
        try{
            sh '''
            #!/usr/bin/env bash
            ins_id=`cat ${WORKSPACE}/instance_id.txt`
            $aws ec2 terminate-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m
            '''
        }catch(err){
            echo err.getMessage()   
        }
    }

    stage('archiveArtifacts') {
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
    }

    // TODO: Enhance the email inform
    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="${default_mail}"
        }

        if (fileExists("${WORKSPACE}/${_target}/inductor_log/guitly_commit.log") == true){
            emailext(
                subject: "Torchinductor-${env._backend}-${env._suite}-${env._model}-${env._test_mode}-${env._precision}-${env._shape}-${env._WAPPER}-${env._scenario}-${env._threads}-${env._kind}-guilty_commit_Report(AWS)_${env._target}",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*_guilty_commit.log",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build succeed, please double check in ${BUILD_URL}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Torchinductor-${env._backend}-${env._suite}-${env._model}-${env._test_mode}-${env._precision}-${env._shape}-${env._WAPPER}-${env._scenario}-${env._threads}-${env._kind}-guilty_commit_Report(AWS)_${env._target}",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}
