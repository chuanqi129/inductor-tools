NODE_LABEL = 'mlp-validate-icx24-ubuntu'
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

target_job = 'inductor_aws_regular_cppwrapper'
if ('target_job' in params) {
    echo "target_job in params"
    if (params.target_job != '') {
        target_job = params.target_job
    }
}
echo "target_job: $target_job"

target_job_selector = 'lastSuccessfulBuild'
if ('target_job_selector' in params) {
    echo "target_job_selector in params"
    if (params.target_job_selector != '') {
        target_job_selector = params.target_job_selector
    }
}
echo "target_job_selector: $target_job_selector"

refer_job = 'inductor_aws_regular_dashboard'
if ('refer_job' in params) {
    echo "refer_job in params"
    if (params.refer_job != '') {
        refer_job = params.refer_job
    }
}
echo "refer_job: $refer_job"

refer_job_selector = 'lastSuccessfulBuild'
if ('refer_job_selector' in params) {
    echo "refer_job_selector in params"
    if (params.refer_job_selector != '') {
        trefer_job_selector = params.refer_job_selector
    }
}
echo "refer_job_selector: $refer_job_selector"

env._target_job = "$target_job"
env._target_sc = "$target_job_selector"
env._refer_job = "$refer_job"
env._refer_sc = "$refer_job_selector"

def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        cd ${WORKSPACE} && sudo rm -rf *
        '''
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

node(NODE_LABEL){
    stage("prepare"){
        echo 'prepare......'
        cleanup()
        deleteDir()
        checkout scm   
    }
    stage("copy"){
        copyArtifacts(
            projectName: "${target_job}",
            selector: specific("${target_job_selector}"),
            fingerprintArtifacts: true
        )          
        sh '''
        #!/usr/bin/env bash
        cd ${WORKSPACE} && mkdir -p target_${_target_sc} && mv inductor_log target_${_target_sc}
        '''
        copyArtifacts(
            projectName: "${refer_job}",
            selector: specific("${refer_job_selector}"),
            fingerprintArtifacts: true
        )            
        sh '''
        #!/usr/bin/env bash
        cd ${WORKSPACE} && mkdir -p refer_${_refer_sc} && mv inductor_log refer_${_refer_sc}
        '''        
    }
    stage("report"){
        sh '''
        #!/usr/bin/env bash
        cp scripts/modelbench/report.py ${WORKSPACE} && python report.py -r refer_${_refer_sc} -t target_${_target_sc} -m all --md_off
        mv target_${_target_sc}/*.xlsx ./ && mv target_${_target_sc}/*.html ./ && rm -rf refer_${_refer_sc} && rn -rf target_${_target_sc}
        '''
        archiveArtifacts  "*.xlsx, *.html"
    }

    stage("Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/inductor_model_bench.html") == true){
            emailext(
                subject: "Torchinductor-${env._target_job}-${env._refer_job}-result-compare-report",
                mimeType: "text/html",
                attachmentsPattern: "**/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="inductor_model_bench.html"}'
            )
        }else{
            emailext(
                subject: "Failure occurs in inductor job compare",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}
