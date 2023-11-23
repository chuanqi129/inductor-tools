NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

debug = ''
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

inductor_tools_branch = ''
if ('inductor_tools_branch' in params) {
    echo "inductor_tools_branch in params"
    if (params.inductor_tools_branch != '') {
        inductor_tools_branch = params.inductor_tools_branch
    }
}
echo "inductor_tools_branch: $inductor_tools_branch"

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
                pip install jinja2
                cp scripts/ci-track/query_issue.py ./
                cp scripts/ci-track/template.html ./
                python3 query_issue.py
            '''
        }
    }
    stage("artifact and email html result") {
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/index.html") == true){
            archiveArtifacts "index.html"
            emailext(
                subject: "Pytorch Inductor Github Issue Least recently updated Track",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="index.html"}'
            )
        }
    } 
}