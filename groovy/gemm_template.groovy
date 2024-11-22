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

def cleanup(){
    try {
        retry(3){
            sh'''
                #!/usr/bin/env bash
                docker_ps=`docker ps -a -q`
                if [ -n "${docker_ps}" ];then
                    docker stop ${docker_ps}
                fi
                docker container prune -f
                docker system prune -f

                docker pull ${BASE_IMAGE}
                docker run -t \
                    -u root \
                    -v ${WORKSPACE}:/root/workspace \
                    --privileged \
                    ${BASE_IMAGE} /bin/bash -c "chmod -R 777 /root/workspace"
            '''
        }
        deleteDir()
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

def pruneOldImage(){
    sh '''
        #!/usr/bin/env bash
        old_image_id=`docker images | grep pt_inductor | awk '{print $3}'`
        old_image=`echo $old_image_id | awk '{print $1}'`
        if [ -n "${old_image}" ]; then
            docker rmi -f $old_image
        fi
        docker system prune -f
    '''
}

node(us_node){
    stage("prepare torch repo"){
        try {
            if  ("${report_only}" == "false") {
                deleteDir()
                sh'''
                    #!/usr/bin/env bash
                    echo "Job URL: ${BUILD_URL}/console .<br>" | tee ${WORKSPACE}/torch_clone.log
                    if [ "${TORCH_COMMIT}" == "nightly" ];then
                        # clone pytorch repo
                        cd ${WORKSPACE}
                        git clone -b ${TORCH_COMMIT} ${TORCH_REPO}
                        cd pytorch
                        main_commit=`git log -n 1 --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1`
                        git checkout ${main_commit} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                        result=${PIPESTATUS[0]}
                    else
                        # clone pytorch repo
                        cd ${WORKSPACE}
                        git clone ${TORCH_REPO}
                        cd pytorch
                        git checkout ${TORCH_COMMIT} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                        result=${PIPESTATUS[0]}
                    fi
                    if [ "${result}" = "0" ]; then
                        echo "<br>[INFO] Torch repo and commit is correct.<br>" | tee -a ${WORKSPACE}/torch_clone.log
                    else
                        echo "<br>[ERROR] Torch repo and commit is wrong!<br>" | tee -a ${WORKSPACE}/torch_clone.log
                        exit 1
                    fi
                    commit_date=`git log -n 1 --format="%cs"`
                    bref_commit=`git rev-parse --short HEAD`
                    DOCKER_TAG="${commit_date}_${bref_commit}"
                    echo "${DOCKER_TAG}" > ${WORKSPACE}/docker_image_tag.log
                '''
                if (fileExists("${WORKSPACE}/docker_image_tag.log")) {
                    stash includes: 'docker_image_tag.log', name: 'docker_image_tag'
                    archiveArtifacts  "docker_image_tag.log"
                }
            }
        } catch (Exception e) {
            emailext(
                subject: "Inductor TAS pipeline Pre-Check failed",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: "$mail_list",
                body: '${FILE, path="torch_clone.log"}'
            )
            throw e
        }
    }
}

node(NODE_LABEL){
    stage("prepare"){
        println('prepare......')
        // TODO: implement report_only logic
        if  ("${report_only}" == "false") {
            cleanup()
            pruneOldImage()
            retry(3){
                sleep(60)
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                    extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
            unstash 'docker_image_tag'
            sh'''
                #!/usr/bin/env bash
                mkdir -p ${WORKSPACE}/${LOG_DIR}
                mv docker_image_tag.log ${WORKSPACE}/${LOG_DIR}
            '''
        } else {
            retry(3){
                sleep(60)
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                    extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
        }
    }

    stage("trigger inductor images job"){
        try {
            if  ("${report_only}" == "false") {
                if ("${build_image}" == "true") {
                    retry(3){
                        sleep(60)
                        def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/${LOG_DIR}/docker_image_tag.log''').toString().trim().replaceAll("\n","")
                        def image_build_job = build job: 'inductor_images_local_py310', propagate: false, parameters: [             
                            [$class: 'StringParameterValue', name: 'PT_REPO', value: "${TORCH_REPO}"],
                            [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${TORCH_COMMIT}"],
                            [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
                            [$class: 'StringParameterValue', name: 'HF_TOKEN', value: "${HF_TOKEN}"],
                        ]
                        def buildStatus = image_build_job.getResult()
                        def cur_job_url = image_build_job.getAbsoluteUrl()
                        if (buildStatus == hudson.model.Result.FAILURE) {
                            sh'''
                                echo "[FAILED] Docker image build Job URL: ${cur_job_url}/console .<br>" | tee ${WORKSPACE}/image_build.log
                            '''
                            throw new Exception("Docker image build job failed")
                        }
                        sh'''
                            #!/usr/bin/env bash
                            docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
                            docker pull ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
                        '''
                    }
                }
                if (fileExists("${WORKSPACE}/${LOG_DIR}/docker_image_tag.log")) {
                    archiveArtifacts  "${LOG_DIR}/docker_image_tag.log"
                }
            }
        } catch (Exception e) {
            sh'''
                #!/usr/bin/env bash
                set -ex
                echo "Job URL: ${BUILD_URL}/console .<br>" | tee -a ${WORKSPACE}/image_build.log
            '''
            emailext(
                subject: "Inductor TAS pipeline Pre-Check failed",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: "$mail_list",
                body: '${FILE, path="image_build.log"}'
            )
            archiveArtifacts "image_build.log"
            throw e
        }
    }

    stage("prepare container"){
        if  ("${report_only}" == "false") {
            withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                sh'''
                    #!/usr/bin/env bash
                    docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
                    docker pull ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
                '''
            }
        }
    }

    stage("benchmark") {
        sh '''
            #!/usr/bin/env bash
            docker_image_tag=`cat ${LOG_DIR}/docker_image_tag.log`
            docker run -tid --name inductor_test \
                --privileged \
                --env https_proxy=${https_proxy} \
                --env http_proxy=${http_proxy} \
                --env HF_HUB_TOKEN=$HF_TOKEN \
                --net host --shm-size 1G \
                -v ~/.cache:/root/.cache \
                -v ${WORKSPACE}/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} \
                ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
            docker exec -i inductor_test bash -c "bash version_collect.sh ${LOG_DIR} $DYNAMO_BENCH"
            docker exec -i inductor_test bash -c "python -m pytest -v test/inductor/test_mkl_verbose.py 2>&1 | tee ${LOG_DIR}/gemm_ut.log"
            # docker exec -i inductor_test bash -c "python -m pytest -v test/inductor/test_cpu_select_algorithm.py 2>&1 | tee ${LOG_DIR}/gemm_ut.log"
            docker exec -i inductor_test bash -c "chmod 777 -R /workspace/pytorch/${LOG_DIR}"
        '''
    }

    stage("stop docker") {
        sh '''
            #!/usr/bin/env bash
            docker_ps=`docker ps -a -q`
            if [ -n "${docker_ps}" ];then
                docker stop ${docker_ps}
            fi
        '''
    }

}
