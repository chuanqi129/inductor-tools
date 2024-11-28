env.DOCKER_IMAGE_NAMESPACE = 'gar-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE = 'gar-registry.caas.intel.com/pytorch/pt_inductor:ubuntu_22.04'
env.LOG_DIR = 'gemm_template_log'
if (env.NODE_LABEL == "0") {
    if (env.precision == "float32") {
        env.NODE_LABEL = "inductor-icx-local-tas"
    } else if (env.precision == 'amp') {
        env.NODE_LABEL = "inductor-gnr-local-tas-sh"
    }
}

mail_list = params.mail_list ?: "lifeng.a.wang@intel.com"


node(NODE_LABEL){
    stage("benchmark") {
        sh '''
        echo 'TTTTTTTTTTTTTTTTTTTTTTTT' > ${LOG_DIR}/gemm_ut.log
        '''
        if (fileExists("${WORKSPACE}/${LOG_DIR}/gemm_ut.log")) {
                archiveArtifacts  "${LOG_DIR}/gemm_ut.log"
            }
    }

    stage("Sent Email"){
        if (fileExists("${WORKSPACE}/${LOG_DIR}/gemm_ut.log") == true){
            emailext(
                subject: "GEMM Template Weekly Test Report",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: "$mail_list",
                body: '${FILE,path="${env.LOG_DIR}/gemm_ut.log" lines=1 start=last}'
            )
        }else{
            emailext(
                subject: "GEMM Template Weekly Test Failed",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: "$mail_list",
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
        
    }

}
