REMOTE = 'True'
if ('REMOTE' in params) {
    echo "REMOTE in params"
    if (params.REMOTE != '') {
        REMOTE = params.REMOTE
    }
}

REMOTE_NODE_LABEL = 'AWS_SSO_REMOTE'
if ('REMOTE_NODE_LABEL' in params) {
    echo "REMOTE_NODE_LABEL in params"
    if (params.REMOTE_NODE_LABEL != '') {
        REMOTE_NODE_LABEL = params.REMOTE_NODE_LABEL
    }
}

NODE_LABEL = 'yudongsi-mlt-ace'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}

debug = 'True'
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}

debug_mail = 'yudong.si@intel.com'
if ('debug_mail' in params) {
    echo "debug_mail in params"
    if (params.debug_mail != '') {
        debug_mail = params.debug_mail
    }
}

aws = '/home2/yudongsi/.local/bin/aws'
if ('aws' in aws) {
    echo "aws in params"
    if (params.aws != '') {
        aws = params.aws
    }
}

firefox = '/home2/yudongsi/workspace/firefox/firefox'
if ('firefox' in params) {
    echo "firefox in params"
    if (params.firefox != '') {
        firefox = params.firefox
    }
}

gd = '/home2/yudongsi/workspace/geckodriver'
if ('gd' in params) {
    echo "gd in params"
    if (params.gd != '') {
        gd = params.gd
    }
}

iap_credential = 'yudongsi_iap'
if ('iap_credential' in params) {
    echo "iap_credential in params"
    if (params.iap_credential != '') {
        iap_credential = params.iap_credential
    }
}

REFRESH_NODE_LABEL = "$NODE_LABEL"
refresh_aws = "$aws"
if ("$REMOTE" == "True") {
    echo "Use remote refresh method"
    REFRESH_NODE_LABEL = "$REMOTE_NODE_LABEL"
    refresh_aws = "/localdisk/chuanqiw/aws-cli/v2/current/bin/aws"
    firefox = "/localdisk/chuanqiw/firefox/firefox"
    gd = "/localdisk/chuanqiw/geckodriver"
}

echo "REFRESH_NODE_LABEL: $REFRESH_NODE_LABEL"
echo "debug: $debug"
echo "debug_mail: $debug_mail"
echo "refresh_aws: $refresh_aws"
echo "firefox: $firefox"
echo "gd: $gd"
echo "iap_credential: $iap_credential"

env._AWS = "$aws"
env._REFRESH_AWS = "$refresh_aws"
env._FF = "$firefox"
env._GD = "$gd"
env._REMOTE = "$REMOTE"

node(REFRESH_NODE_LABEL){
    stage("AWS SSO Refresh")
    {
        deleteDir()
        checkout scm
        try{
            withCredentials([usernamePassword(credentialsId: "$iap_credential", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')])
            {        
                sh '''
                #!/usr/bin/env bash
                cd scripts/aws && bash refresh.sh ${_REFRESH_AWS} ${_FF} ${_GD} $USERNAME $PASSWORD $REMOTE
                '''
            }
            if ("$REMOTE" == "True") {
                echo "Hard code copy sso configure back!"
                sh '''
                #!/bin/bash
                scp -r ~/.aws yudongsi@mlt-ace:~/
                '''
            }
        }catch(err){
            echo err.getMessage()
            currentBuild.result = "FAILURE"   
        }
    }
}

node(NODE_LABEL){
    stage("Sanity test of the AWS SSO configure"){
        echo "All running instance.........."
        sh '''
        #!/usr/bin/env bash
        ${_AWS} ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId]' --filters Name=instance-state-name,Values=running --output text --profile pytorch
        '''
    }
    stage("Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="yudong.si@intel.com;Chuanqi.Wang@intel.com;shufan.wu@intel.com;diwei.sun@intel.com"
        }
        if (currentBuild.currentResult == 'FAILURE'){
            emailext(
                subject: "AWS SSO Refresh JOB Build Failure",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Please double check in ${BUILD_URL}'
            )
        }
    }//email    
}