GIT_CREDENTIAL = "ESI-SYD-Github-Credentials"
NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

node(NODE_LABEL){
    stage("clean directory and prepare script"){
        echo 'clean directory and prepare script......'
        deleteDir()
        checkout scm
    }
    stage("run script to get refresh"){
        retry(3){
            echo 'run script to get refresh......'
            sh '''
                set +e
                python3 scripts/ci-track/ci_track.py
            '''
        }
    }
    stage("artifact and email html result") {
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiangdong.zeng@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/CI_JOB_FAILURE_TRACK.html") == true){
            archiveArtifacts "CI_JOB_FAILURE_TRACK.html"
            emailext(
                subject: "Torchinductor PRs CI Job Failure Track Report",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="CI_JOB_FAILURE_TRACK.html"}'
            )
        }
    } 
}
