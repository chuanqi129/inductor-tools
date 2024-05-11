NODE_LABEL = 'mlp-spr-04.sh.intel.com'
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

cppwp_gm = 'False'
if ('cppwp_gm' in params) {
    echo "cppwp_gm in params"
    if (params.cppwp_gm != '') {
        cppwp_gm = params.cppwp_gm
    }
}
echo "cppwp_gm: $cppwp_gm"

mt_start = '0.04'
if ('mt_start' in params) {
    echo "mt_start in params"
    if (params.mt_start != '') {
        mt_start = params.mt_start
    }
}
echo "mt_start: $mt_start"

mt_end = '1.5'
if ('mt_end' in params) {
    echo "mt_end in params"
    if (params.mt_end != '') {
        mt_end = params.mt_end
    }
}
echo "mt_end: $mt_end"

st_start = '0.04'
if ('st_start' in params) {
    echo "st_start in params"
    if (params.st_start != '') {
        st_start = params.st_start
    }
}
echo "st_start: $st_start"

st_end = '5'
if ('st_end' in params) {
    echo "st_end in params"
    if (params.st_end != '') {
        st_end = params.st_end
    }
}
echo "st_end: $st_end"

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
        refer_job_selector = params.refer_job_selector
    }
}
echo "refer_job_selector: $refer_job_selector"

test_mode = 'inference'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"

if (test_mode == "training_full") {
    infer_or_train = "training"
} else {
    infer_or_train = test_mode
}

suite= 'all'
if ('suite' in params) {
    echo "suite in params"
    if (params.suite != '') {
        suite = params.suite
    }
}
echo "suite: $suite"

env._target_job = "$target_job"
env._target_sc = "$target_job_selector"
env._refer_job = "$refer_job"
env._refer_sc = "$refer_job_selector"
env._test_mode = "$test_mode"

env._cppwp_gm = "$cppwp_gm"
env._mt_start = "$mt_start"
env._mt_end = "$mt_end"
env._st_start = "$st_start"
env._st_end = "$st_end"
env._suite = "$suite"
env._infer_or_train = "$infer_or_train"

env._NODE = "$NODE_LABEL"

def getUpstreamParameters(String job_name, String job_id) {
    def params = [:]
    try {
        def upstream_job = Jenkins.getInstance().getItemByFullName(job_name).getLastSuccessfulBuild()
        if (job_id != "lastSuccessfulBuild") {
            upstream_job = Jenkins.getInstance().getItemByFullName(job_name).getBuildByNumber(job_id.toInteger())
        }
        def param_list = upstream_job.actions.find{ a -> a instanceof ParametersAction }?.parameters

        param_list.each { p ->
              params[p.name] = p.value
        }
    } catch(NullPointerException ex) {
        echo "WARNING: this script is expected to be triggered by upstream_job."
    }
    return params
}

if ("${debug}" == "true"){
    maillist="${debug_mail}"
}else{
    maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiangdong.zeng@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
}

node(NODE_LABEL){
    stage("prepare"){
        echo 'prepare......'
        deleteDir()
        checkout scm   
    }
    if ((_target_job == _refer_job) && (_target_sc == _refer_sc)) {
        stage("email"){
            emailext(
                subject: "Failure occurs in inductor job compare",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Target job and Reference job are the same, please double check in ${BUILD_URL}'
            )
        }
    } else {
        stage("copy"){
            copyArtifacts(
                projectName: "${target_job}",
                selector: specific("${target_job_selector}"),
                fingerprintArtifacts: true
            )          
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && rm inductor_log/*.html && rm inductor_log/*.xlsx && mkdir -p ${_target_job}_${_target_sc} && mv inductor_log ${_target_job}_${_target_sc}
            '''
            copyArtifacts(
                projectName: "${refer_job}",
                selector: specific("${refer_job_selector}"),
                fingerprintArtifacts: true
            )            
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE} && rm inductor_log/*.html && rm inductor_log/*.xlsx && mkdir -p ${_refer_job}_${_refer_sc} && mv inductor_log ${_refer_job}_${_refer_sc}
            '''        
        }
        stage("report"){
            def params = getUpstreamParameters(_target_job, _target_sc)
            env.shape = params.get('shape')
            env.wrapper = params.get('WRAPPER')
            env.torch_repo = params.get('TORCH_REPO')
            env.torch_branch = params.get('TORCH_BRANCH')
            env._suite = params.get('suite')
            env._precision = params.get('precision')
            env.backend = params.get('backend')

            def ref_params = getUpstreamParameters(_refer_job, _refer_sc)
            env.ref_backend = ref_params.get('backend')
            sh '''
            #!/usr/bin/env bash
            if [ ${_NODE} == 'mlp-spr-04.sh.intel.com' ];then
                source activate pytorch
            fi
            # Install dependencies
            pip install scipy datacompy PyGithub styleframe pandas bs4 requests
            cp scripts/modelbench/report.py ${WORKSPACE}
            if [ ${_cppwp_gm} == 'True' ];then
                python report.py -r ${_refer_job}_${_refer_sc} -t ${_target_job}_${_target_sc} -m all --md_off --url ${BUILD_URL} --precision ${_precision} --cppwrapper_gm --mt_interval_start ${_mt_start} --mt_interval_end ${_mt_end} --st_interval_start ${_st_start} --st_interval_end ${_st_end} --suite ${_suite} --infer_or_train ${_infer_or_train} --shape ${shape} --wrapper ${wrapper} --torch_repo ${torch_repo} --torch_branch ${torch_branch}  --backend ${backend} --threshold ${threshold} --ref_backend ${ref_backend} 
            else
                python report.py -r ${_refer_job}_${_refer_sc} -t ${_target_job}_${_target_sc} -m all --md_off --url ${BUILD_URL} --precision ${_precision} --suite ${_suite} --infer_or_train ${_infer_or_train} --shape ${shape} --wrapper ${wrapper} --shape ${shape} --wrapper ${wrapper} --torch_repo ${torch_repo} --torch_branch ${torch_branch} --backend ${backend} --threshold ${threshold} --ref_backend ${ref_backend} 
            fi
            mv ${_target_job}_${_target_sc}/inductor_log/*.xlsx ./ && mv ${_target_job}_${_target_sc}/inductor_log/*.html ./
            '''
            archiveArtifacts  "*.xlsx, *.html"
            if (fileExists("${WORKSPACE}/guilty_commit_search_model_list.csv")) {
                archiveArtifacts  "guilty_commit_search*"
            }
            if (fileExists("${WORKSPACE}/all_model_list.csv")) {
                archiveArtifacts  "all_model_list.csv"
            }
        }

        stage("Email"){
            if (fileExists("${WORKSPACE}/inductor_model_bench.html") == true){
                emailext(
                    subject: "[report-compare]-${env._target_job}-${env._refer_job}",
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
}
