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

instance_ids = 'i-039458152f180ba94'
if ('instance_ids' in params) {
    echo "instance_ids in params"
    if (params.instance_ids != '') {
        instance_ids = params.instance_ids
    }
}
echo "instance_ids: $instance_ids"

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

env._aws_id = "$instance_ids"
env._reference = "$refer_build"
env._gh_token = "$gh_token"
env._target = new Date().format('yyyy_MM_dd')
println(env._target)

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
            ssh ubuntu@${current_ip} "nohup bash entrance.sh ${_target} &>/dev/null &" &
            '''
        }
    }
    stage("acquire logs"){
        sh '''
        #!/usr/bin/env bash
        set +e
        current_ip=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
        for t in {1..25}
        do
            ssh ubuntu@${current_ip} "test -f /home/ubuntu/docker/finished_float32_inference_static.txt"
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
                if [ $t -eq 22 ]; then
                    echo restart instance now...
                    $aws ec2 stop-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
                    $aws ec2 start-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
                    current_ip=`$aws ec2 describe-instances --instance-ids ${_aws_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
                    echo update ip $current_ip
                    ssh -o StrictHostKeyChecking=no ubuntu@${current_ip} "pwd"
                fi                
            fi
        done
        '''

    }
    stage("stop instance")
    {
        sh '''
        #!/usr/bin/env bash
        $aws ec2 stop-instances --instance-ids ${_aws_id} --profile pytorch && sleep 2m
        '''
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
            cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer -t ${_target} -m all --gh_token ${_gh_token} && rm -rf refer
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
            cd ${WORKSPACE} && mv ${WORKSPACE}/${_target}/inductor_log/ ./ && rm -rf ${_target}
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
