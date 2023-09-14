NODE_LABEL = 'diweisun-mlt-ace'
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

aws_hostname = 'ec2-44-200-107-16.compute-1.amazonaws.com'
if ('aws_hostname' in params) {
    echo "aws_hostname in params"
    if (params.aws_hostname != '') {
        aws_hostname = params.aws_hostname
    }
}
echo "aws_hostname: $aws_hostname"

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

IPEX_REPO = 'https://github.com/intel/intel-extension-for-pytorch.git'
if ('IPEX_REPO' in params) {
    echo "IPEX_REPO in params"
    if (params.IPEX_REPO != '') {
        IPEX_REPO = params.IPEX_REPO
    }
}
echo "IPEX_REPO: $IPEX_REPO"

IPEX_BRANCH= 'master'
if ('IPEX_BRANCH' in params) {
    echo "IPEX_BRANCH in params"
    if (params.IPEX_BRANCH != '') {
        IPEX_BRANCH = params.IPEX_BRANCH
    }
}
echo "IPEX_BRANCH: $IPEX_BRANCH"

IPEX_COMMIT= 'master'
if ('IPEX_COMMIT' in params) {
    echo "IPEX_COMMIT in params"
    if (params.IPEX_COMMIT != '') {
        IPEX_COMMIT = params.IPEX_COMMIT
    }
}
echo "IPEX_COMMIT: $IPEX_COMMIT"

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

FUSION_PATH= 'torchscript'
if ('FUSION_PATH' in params) {
    echo "FUSION_PATH in params"
    if (params.FUSION_PATH != '') {
        FUSION_PATH = params.FUSION_PATH
    }
}
echo "FUSION_PATH: $FUSION_PATH"

aws_id= 'i-009c3b5297e7029ad'
if ('aws_id' in params) {
    echo "aws_id in params"
    if (params.aws_id != '') {
        FUSION_PATH = params.aws_id
    }
}
echo "aws_id: $aws_id"

env._name = "$aws_hostname"
env._reference = "$refer_build"
env._gh_token = "$gh_token"
env._target = new Date().format('yyyy_MM_dd')

env._precision = "$precision"
env._test_mode = "$test_mode"
env._shape="$shape"
env._TORCH_REPO = "$TORCH_REPO"
env._TORCH_BRANCH = "$TORCH_BRANCH"
env._TORCH_COMMIT = "$TORCH_COMMIT"
env._IPEX_REPO = "$IPEX_REPO"
env._IPEX_BRANCH = "$IPEX_BRANCH"
env._IPEX_COMMIT = "$IPEX_COMMIT"
env._DYNAMO_BENCH = "$DYNAMO_BENCH"

env._AUDIO = "$AUDIO"
env._TEXT = "$TEXT"
env._VISION = "$VISION"
env._DATA = "$DATA"
env._TORCH_BENCH = "$TORCH_BENCH"
env._THREADS = "$THREADS"
env._FUSION_PATH = "$FUSION_PATH"
env._aws_id="$aws_id"
println(env._target)

node(NODE_LABEL){
    stage("Instance Start") {
        sh '''
        /home2/diweisun/.local/bin/aws ec2 start-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
        init_ip=`/home2/diweisun/.local/bin/aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        echo init_ip is $init_ip
        ssh -o StrictHostKeyChecking=no ubuntu@${init_ip} "pwd"
        '''
    }
    stage("prepare scripts") {
        deleteDir()
        checkout scm
        retry(3){
            sh '''
            #!/usr/bin/env bash
            _name=`/home2/diweisun/.local/bin/aws/aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            cd $HOME && cat .ssh/config
            scp ${WORKSPACE}/scripts/modelbench/entrance.sh ubuntu@${_name}:/home/ubuntu
            scp ${WORKSPACE}/docker/Dockerfile.ipex ubuntu@${_name}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/launch.sh ubuntu@${_name}:/home/ubuntu/docker
            scp ${WORKSPACE}/scripts/modelbench/ipex_test.sh ubuntu@${_name}:/home/ubuntu/docker
            '''
        }
    }

    stage("launch benchmark") {
        retry(3){
            sh '''
            #!/usr/bin/env bash
            _name=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            ssh ubuntu@${_name} "nohup bash entrance.sh ${_target} ${_precision} ${_test_mode} ${_shape} ${_TORCH_REPO} ${_TORCH_BRANCH} ${_TORCH_COMMIT} ${_DYNAMO_BENCH} ${_IPEX_REPO} ${_IPEX_BRANCH} ${_IPEX_COMMIT}  ${_AUDIO} ${_TEXT} ${_VISION} ${_DATA} ${_TORCH_BENCH} ${_THREADS} ${_FUSION_PATH} > entrance.log 2>&1 &" &
            '''
        }
    }

    stage("acquire logs"){
        retry(3){
            sh '''
            #!/usr/bin/env bash
            set +e
            _name=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            for t in {1..25}
            do
                ssh ubuntu@${_name} "test -f /home/ubuntu/docker/finished_${_precision}_${_test_mode}_${_shape}.txt"
                if [ $? -eq 0 ]; then
                    if [ -d ${WORKSPACE}/${_target} ]; then
                        rm -rf ${WORKSPACE}/${_target}
                    fi
                    mkdir -p ${WORKSPACE}/${_target}
                    scp -r ubuntu@${_name}:/home/ubuntu/docker/ipex_log ${WORKSPACE}/${_target}
                    break
                else
                    sleep 1h
                    echo $t
                fi
            done
            '''
        }
    }

    stage("generate report"){
        retry(3){
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    fingerprintArtifacts: true
                )             
            sh '''
            #!/usr/bin/env bash      
            source activate ipex_report  
            cd ${WORKSPACE} && mkdir -p refer && cp -r ipex_log refer && rm -rf ipex_log
            cp ${WORKSPACE}/scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target} -m all && rm -rf refer 
            '''
            }else{
                sh '''
                #!/usr/bin/env bash
                source activate ipex_report
                cd ${WORKSPACE} && cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -t ${_target} -m all
                '''
            }
        }
    }


    stage('archiveArtifacts') {
            sh '''
            #!/usr/bin/env bash
            echo ${_target}
            cp -r  ${WORKSPACE}/${_target} $HOME/ipex_dashboard
            cp -r  ${WORKSPACE}/${_target}/ipex_log ${WORKSPACE} && rm -rf ${WORKSPACE}/${_target}
            
            '''        
        archiveArtifacts artifacts: "**/ipex_log/**", fingerprint: true
    }
    stage("Instance ShutDown") {
        sh '''
            /home2/diweisun/.local/bin/aws ec2 stop-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
        '''
    }
    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="diwei.sun@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com;jiayi.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/ipex_log/ipex_model_bench.html") == true){
            emailext(
                subject: "IPEX as TorchDynamo Regularly Benchmark Report (AWS)",
                mimeType: "text/html",
                attachmentsPattern: "**/ipex_log/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="ipex_log/ipex_model_bench.html"}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Torchinductor Regularly Benchmark (AWS)",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email

}
