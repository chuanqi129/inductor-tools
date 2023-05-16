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

aws_hostname = 'ec2-34-207-213-120.compute-1.amazonaws.com'
if ('aws_hostname' in params) {
    echo "aws_hostname in params"
    if (params.aws_hostname != '') {
        aws_hostname = params.aws_hostname
    }
}
echo "aws_hostname: $aws_hostname"

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

env._name = "$aws_hostname"
env._test_mode = "$test_mode"
env._precision = "$precision"
env._shape = "$shape"
env._target = new Date().format('yyyy_MM_dd')
env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH"

node(NODE_LABEL){
    stage("prepare scripts") {
        deleteDir()
        checkout scm
        retry(3){
            sh '''
            #!/usr/bin/env bash
            cd $HOME && cat .ssh/config
            scp ${WORKSPACE}/scripts/modelbench/entrance.sh ubuntu@${_name}:/home/ubuntu
            scp ${WORKSPACE}/docker/Dockerfile ubuntu@${_name}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/launch.sh ubuntu@${_name}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_test.sh ubuntu@${_name}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/inductor_train.sh ubuntu@${_name}:/home/ubuntu/docker
            '''
        }
    }    
    stage("launch benchmark") {
        retry(3){
            sh '''
            #!/usr/bin/env bash
            ssh ubuntu@${_name} "nohup bash entrance.sh ${_target} ${_precision} ${_test_mode} ${_shape} ${_TORCH_REPO} ${_TORCH_BRANCH} ${_TORCH_COMMIT} ${_DYNAMO_BENCH} &>/dev/null &" &
            '''
        }
        sh '''
        #!/usr/bin/env bash
        set +e
        for t in {1..25}
        do
            ssh ubuntu@${_name} "test -f /home/ubuntu/docker/finished_${_precision}_${_test_mode}_${_shape}.txt"
            if [ $? -eq 0 ]; then
                if [ -d ${WORKSPACE}/${_target}_odm ]; then
                    rm -rf ${WORKSPACE}/${_target}_odm
                fi
                mkdir -p ${WORKSPACE}/${_target}_odm
                scp -r ubuntu@${_name}:/home/ubuntu/docker/inductor_log ${WORKSPACE}/${_target}_odm
                break
            else
                sleep 1h
                echo $t
            fi
        done
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
                cd ${WORKSPACE} && mkdir -p refer && cp -r inductor_log refer && rm -rf inductor_log
                cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target}_odm -m all --md_off --precision ${_precision} && rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && python report.py -t ${_target}_odm -m all --md_off --precision ${_precision}
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
                cp scripts/modelbench/report_train.py ${WORKSPACE} && python report_train.py -r refer -t ${_target}_odm && rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && python report_train.py -t ${_target}_odm
                '''
            }
        }
    }    

    stage('archiveArtifacts') {
        if ("${test_mode}" == "inference")
        {
            sh '''
            #!/usr/bin/env bash
            cp -r  ${WORKSPACE}/${_target}_odm $HOME/inductor_dashboard
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            cp -r  ${WORKSPACE}/${_target}_odm $HOME/inductor_dashboard/Train
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
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-Report(AWS)",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-(AWS)",
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
                    subject: "Torchinductor-${env._test_mode}-${env._precision}-${env._shape}-Report(AWS)",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_training_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor Training Benchmark (AWS)",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }           
        }//training
    }//email
}
