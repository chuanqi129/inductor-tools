env.instance_name = 'spr-hf-oob'
env.target = new Date().format('yyyy_MM_dd')
env.LOG_DIR = 'hf_oob_log'

env.terminate_instance = 'False'
if ('terminate_instance' in params) {
    echo "terminate_instance in params"
    if (params.terminate_instance != '') {
        env.terminate_instance = params.terminate_instance
    }
}
echo "terminate_instance: $terminate_instance"

def getGroovyEnv() {
    def env_groovy = ''
    env.getEnvironment().each {
        name, value ->
            if(value){
                env_groovy = env_groovy + "${name}=${value.trim().replaceAll("\n", "")}\n"
            }else{
                env_groovy = env_groovy + "${name}=NULL\n"
            }
    }
    return env_groovy
}

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
                bash find_instance.sh ${instance_name} ${instance_ids} 2>&1 | tee ${WORKSPACE}/instance_id.txt
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
            echo ${instance_ids} > ${WORKSPACE}/instance_id.txt
            echo "ins_id : ${instance_ids}"
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
        def env_groovy = getGroovyEnv()
        writeFile file: "env_groovy.txt", text: env_groovy
        archiveArtifacts  "env_groovy.txt"
        if  ("${report_only}" == "false")
        {
            sh '''
            #!/usr/bin/env bash
            ins_id=`cat ${WORKSPACE}/instance_id.txt`
            current_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            ssh ubuntu@${current_ip} "if [ ! -d /home/ubuntu/docker ]; then mkdir -p /home/ubuntu/docker; fi"
            scp ${WORKSPACE}/scripts/aws/docker_prepare.sh ubuntu@${current_ip}:/home/ubuntu
            ssh ubuntu@${current_ip} "bash docker_prepare.sh"
            scp ${WORKSPACE}/scripts/modelbench/pkill.sh ubuntu@${current_ip}:/home/ubuntu
            scp ${WORKSPACE}/scripts/modelbench/hf_oob_test.sh ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/docker/Dockerfile.hf_oob ubuntu@${current_ip}:/home/ubuntu/docker
            scp ${WORKSPACE}/env_groovy.txt ubuntu@${current_ip}:/home/ubuntu/docker
            ssh ubuntu@${current_ip} "bash pkill.sh"
            ssh ubuntu@${current_ip} "cd /home/ubuntu/docker; nohup bash hf_oob_test.sh &>/dev/null &" &
            '''
        }
    }
    stage("log query") {
        sh '''
        #!/usr/bin/env bash
        set +e
        reboot_time=33
        ins_id=`cat ${WORKSPACE}/instance_id.txt`        
        for t in {1..100}
        do
            current_ip=`$aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text`
            timeout 2m ssh ubuntu@${current_ip} "test -f /home/ubuntu/docker/finished.txt"
            if [ $? -eq 0 ]; then
                if [ -d ${WORKSPACE}/${target} ]; then
                    rm -rf ${WORKSPACE}/${target}
                fi
                mkdir -p ${WORKSPACE}/${target}
                scp -r ubuntu@${current_ip}:/home/ubuntu/docker/${LOG_DIR} ${WORKSPACE}/${target}
                break
            else
                sleep 1h
                echo $t
                if [ $t -eq $reboot_time ]; then
                    echo restart instance now...
                    $aws ec2 stop-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m && $aws ec2 start-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m && current_ip=$($aws ec2 describe-instances --instance-ids ${ins_id} --profile pytorch --query 'Reservations[*].Instances[*].PublicDnsName' --output text) && echo update_ip $current_ip || echo $current_ip
                    ssh -o StrictHostKeyChecking=no ubuntu@${current_ip} "pwd"
                    scp -r ubuntu@${current_ip}:/home/ubuntu/docker/${LOG_DIR} ${WORKSPACE}/${target}
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
            cp -r ${WORKSPACE}/${target} ${WORKSPACE}/raw_log
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
            if [ "$terminate_instance" == "True" ]; then
                $aws ec2 terminate-instances --instance-ids ${ins_id} --profile pytorch && sleep 2m
            fi
            '''
        }catch(err){
            echo err.getMessage()   
        }
    }

    stage("generate report"){
        sh '''
            #!/usr/bin/env bash
            python scripts/modelbench/hf_oob_report.py -t ${target}
        '''
        archiveArtifacts artifacts: "*.csv", fingerprint: true
    }

    stage('archiveArtifacts') {
        // Remove raw log fistly in case inducto_log will be artifact more than 2 times
        sh '''
            #!/usr/bin/env bash
            rm -rf ${WORKSPACE}/raw_log
        '''
        archiveArtifacts artifacts: "**/"+LOG_DIR+"/**", fingerprint: true
    }

    // stage("Sent Email"){
    //    
    // }//email
}
