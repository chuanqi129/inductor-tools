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

aws_hostname = 'ec2-34-207-213-120.compute-1.amazonaws.com'
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

env._name = "$aws_hostname"
env._reference = "$refer_build"
env._gh_token = "$gh_token"
env._target = new Date().format('yyyy_MM_dd')
println(env._target)

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
            '''
        }
    }

    stage("launch benchmark") {
        retry(3){
            sh '''
            #!/usr/bin/env bash
            ssh ubuntu@${_name} "nohup bash entrance.sh ${_target} &>/dev/null &" &
            '''
        }
    }

    stage("acquire logs"){
        retry(3){
            sh '''
            #!/usr/bin/env bash
            set +e
            for t in {1..25}
            do
                ssh ubuntu@${_name} "test -f /home/ubuntu/docker/finished.txt"
                if [ $? -eq 0 ]; then
                    if [ -d ${WORKSPACE}/${_target} ]; then
                        rm -rf ${WORKSPACE}/${_target}
                    fi
                    mkdir -p ${WORKSPACE}/${_target}
                    scp -r ubuntu@${_name}:/home/ubuntu/docker/inductor_log ${WORKSPACE}/${_target}
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
            cd ${WORKSPACE} && mkdir -p refer && cp -r inductor_log refer && rm -rf inductor_log
            cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target} -m all && rm -rf refer --gh_token ${_gh_token}
            '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE} && cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -t ${_target} -m all
                '''
            }
        }
    }

    stage('archiveArtifacts') {
            sh '''
            #!/usr/bin/env bash
            cp -r  ${WORKSPACE}/${_target} $HOME/inductor_dashboard
            '''        
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
    }

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/inductor_log/inductor_model_bench.html") == true){
            emailext(
                subject: "Torchinductor Regularly Benchmark Report (AWS)",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
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
