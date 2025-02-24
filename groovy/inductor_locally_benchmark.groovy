env.backend = 'inductor'
if ('backend' in params) {
    echo "backend in params"
    if (params.backend != '') {
        env.backend = params.backend
    }
}
echo "backend: $backend"

env.dash_board = 'false'
if ('dash_board' in params) {
    echo "dash_board in params"
    if (params.dash_board != '') {
        env.dash_board = params.dash_board
    }
}
echo "dash_board: $dash_board"

env.dashboard_title = 'default'
if ('dashboard_title' in params) {
    echo "dashboard_title in params"
    if (params.dashboard_title != '') {
        env.dashboard_title = params.dashboard_title
    }
}
echo "dashboard_title: $dashboard_title"

env.test_ENV = ''
if ('test_ENV' in params) {
    echo "test_ENV in params"
    if (params.test_ENV != '') {
        env.test_ENV = params.test_ENV
    }
}
echo "test_ENV: $test_ENV"

if (env.test_mode == "training_full") {
    env.infer_or_train = "training"
} else {
    env.infer_or_train = test_mode
}

if ( env.dash_board == "true" ) {
    env.dashboard_args = " --gh_token ${gh_token} --dashboard ${dashboard_title} "
} else {
    env.dashboard_args = " --md_off "
}

env.bench_machine = "Local"
env.target = new Date().format('yyyy_MM_dd')
env.DOCKER_IMAGE_NAMESPACE = 'gar-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE= 'gar-registry.caas.intel.com/pytorch/pt_inductor:ubuntu_22.04'
env.LOG_DIR = 'inductor_log'
env.DYNAMO_BENCH = env.TORCH_COMMIT
env.test_ENV = env.test_ENV.replaceAll(" ", "")
if (env.NODE_LABEL == "0") {
    if (env.precision == "float32") {
        env.NODE_LABEL = "inductor-icx-local-tas"
    } else if (env.precision == 'amp') {
        env.NODE_LABEL = "inductor-gnr-local-tas-sh"
    }
}

if ("${debug}" == "true"){
    maillist="${debug_mail}"
}else{
    maillist="${default_mail}"
}

env.report_prefix = "[Regular Weekly]"
if (env.JOB_NAME == "inductor_locally_benchmark") {
    env.report_prefix = "[TAS]"
}

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
                to: maillist,
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
                to: maillist,
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
        if  ("${report_only}" == "false") {
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
                docker cp scripts/modelbench/inductor_test.sh inductor_test:/workspace/pytorch
                docker cp scripts/modelbench/inductor_train.sh inductor_test:/workspace/pytorch
                docker cp scripts/modelbench/version_collect.sh inductor_test:/workspace/pytorch
                docker cp scripts/install_gcc13.sh inductor_test:/workspace/pytorch
                docker exec -i inductor_test bash -c "bash install_gcc13.sh"
                docker exec -i inductor_test bash -c "bash version_collect.sh ${LOG_DIR} $DYNAMO_BENCH"

                if [ $test_mode == "inference" ]; then
                    docker exec -i inductor_test bash -c "bash inductor_test.sh $THREADS $CHANNELS $precision $shape ${LOG_DIR} $WRAPPER $HF_TOKEN $backend inference $suite ${test_ENV} $extra_param "
                elif [ $test_mode == "training_full" ]; then
                    docker exec -i inductor_test bash -c "bash inductor_test.sh multiple $CHANNELS $precision $shape ${LOG_DIR} $WRAPPER $HF_TOKEN $backend training $suite ${test_ENV} $extra_param"
                elif [ $test_mode == "training" ]; then
                    docker exec -i inductor_test bash -c "bash inductor_train.sh $CHANNELS $precision ${LOG_DIR} ${test_ENV} $extra_param"
                fi
                docker exec -i inductor_test bash -c "chmod 777 -R /workspace/pytorch/${LOG_DIR}"
            '''
        }
    }

    // Add raw log artifact stage in advance to avoid crash in report generate stage
    stage("archive raw test results"){
        if  ("${report_only}" == "false") {
            sh '''
                #!/usr/bin/env bash
                if [ -d ${WORKSPACE}/raw_log ];then
                    rm -rf ${WORKSPACE}/raw_log
                fi
                if [ -d ${WORKSPACE}/${target} ];then
                    rm -rf ${WORKSPACE}/${target}
                fi
                cp -r ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/raw_log
                mkdir ${WORKSPACE}/${target}
                mv ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/${target}/
            '''
        } else {
            sh '''
                #!/usr/bin/env bash
                if [ -d ${WORKSPACE}/${target} ];then
                    rm -rf ${WORKSPACE}/${target}
                fi
                if [ -d ${WORKSPACE}/raw_log ];then
                    rm -rf ${WORKSPACE}/raw_log
                fi
                cp -r ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/raw_log
                rm -f ${WORKSPACE}/${LOG_DIR}/*.xlsx
                rm -f ${WORKSPACE}/${LOG_DIR}/*.html
                mkdir ${WORKSPACE}/${target}
                mv ${WORKSPACE}/${LOG_DIR} ${WORKSPACE}/${target}/
            '''
        }
        archiveArtifacts artifacts: "**/raw_log/**", fingerprint: true
    }

    stage("stop docker") {
        sh'''
            #!/usr/bin/env bash
            docker_ps=`docker ps -a -q`
            if [ -n "${docker_ps}" ];then
                docker stop ${docker_ps}
            fi
        '''
    }

    stage("generate report"){
        if ("${test_mode}" == "inference" || "${test_mode}" == "training_full")
        {
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    fingerprintArtifacts: true,
                    target: "refer",
                )           
                sh '''
                    #!/usr/bin/env bash
                    docker_image_tag=`cat ${WORKSPACE}/raw_log/docker_image_tag.log`
                    cd ${WORKSPACE}
                    if [ "${precision}" == "amp_fp16" ];then
                        export precision='amp'
                    fi
                    if [ "${backend}" == "triton_cpu" ];then
                        export backend='inductor'
                    fi
                    cp scripts/modelbench/report.py ${WORKSPACE}
                    python report.py \
                        -r refer \
                        -t ${target} \
                        -m ${THREADS} \
                        --precision ${precision} \
                        --url ${BUILD_URL} \
                        --image_tag ${docker_image_tag} \
                        --suite ${suite} \
                        --infer_or_train ${infer_or_train} \
                        --shape ${shape} \
                        --wrapper ${WRAPPER} \
                        --torch_repo ${TORCH_REPO} \
                        --backend ${backend} \
                        --ref_backend ${backend} \
                        ${dashboard_args}
                    rm -rf refer
                '''
            }else{
                sh '''
                    #!/usr/bin/env bash
                    docker_image_tag=`cat ${WORKSPACE}/raw_log/docker_image_tag.log`
                    cd ${WORKSPACE}
                    if [ "${precision}" == "amp_fp16" ];then
                        export precision='amp'
                    fi
                    if [ "${backend}" == "triton_cpu" ];then
                        export backend='inductor'
                    fi
                    cp scripts/modelbench/report.py ${WORKSPACE}
                    python report.py \
                        -t ${target} \
                        -m ${THREADS} \
                        --precision ${precision} \
                        --url ${BUILD_URL} \
                        --image_tag ${docker_image_tag} \
                        --suite ${suite} \
                        --infer_or_train ${infer_or_train} \
                        --shape ${shape} \
                        --wrapper ${WRAPPER} \
                        --torch_repo ${TORCH_REPO} \
                        --backend ${backend} \
                        ${dashboard_args}
                '''
            }
        }
        if ("${test_mode}" == "training")
        {
            if(refer_build != '0') {
                copyArtifacts(
                    projectName: currentBuild.projectName,
                    selector: specific("${refer_build}"),
                    fingerprintArtifacts: true
                )
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE}
                mkdir -p refer
                cp -r ${LOG_DIR} refer
                rm -rf ${LOG_DIR}
                cp scripts/modelbench/report_train.py ${WORKSPACE}
                python report_train.py -r refer -t ${target} && rm -rf refer
                '''
            }else{
                sh '''
                #!/usr/bin/env bash
                cd ${WORKSPACE}
                cp scripts/modelbench/report_train.py ${WORKSPACE}
                python report_train.py -t ${target}
                '''
            }
        }
    }    

    stage('archiveArtifacts') {
        // Remove raw log fistly in case inducto_log will be artifact more than 2 times
        sh '''
            #!/usr/bin/env bash
            if [ -d ${WORKSPACE}/raw_log ];then
                rm -rf ${WORKSPACE}/raw_log
            fi
        '''
        if ("${test_mode}" == "inference" || "${test_mode}" == "training_full")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard
            cp -r  ${WORKSPACE}/${target} $HOME/inductor_dashboard
            cd ${WORKSPACE} && mv ${WORKSPACE}/${target}/${LOG_DIR}/ ./&& rm -rf ${target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard/Train
            cp -r  ${WORKSPACE}/${target} $HOME/inductor_dashboard/Train
            cd ${WORKSPACE} && mv ${WORKSPACE}/${target}/${LOG_DIR}/ ./&& rm -rf ${target}
            '''
        }
        archiveArtifacts artifacts: "**/${LOG_DIR}/**", fingerprint: true
        if (fileExists("${WORKSPACE}/guilty_commit_search_model_list.csv")) {
            archiveArtifacts  "guilty_commit_search*"
        }
        if (fileExists("${WORKSPACE}/all_model_list.csv")) {
            archiveArtifacts  "all_model_list.csv"
        }
    }

    stage("Sent Email"){
        if ("${test_mode}" == "inference")
        {
            if (fileExists("${WORKSPACE}/${LOG_DIR}/inductor_model_bench.html") == true){
                emailext(
                    subject: "${env.report_prefix}-Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-Report(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/${LOG_DIR}/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: "\${FILE,path=\"${env.LOG_DIR}/inductor_model_bench.html\"}"
                )
            }else{
                emailext(
                    subject: "${env.report_prefix}-Failure occurs in Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//inference
        if ("${test_mode}" == "training" || "${test_mode}" == "training_full")
        {
            if (fileExists("${WORKSPACE}/${LOG_DIR}/inductor_model_training_bench.html") == true){
                emailext(
                    subject: "${env.report_prefix}-Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-Report(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/${LOG_DIR}/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: "\${FILE,path=\"${env.LOG_DIR}/inductor_model_bench.html\"}"
                )
            }else{
                emailext(
                    subject: "${env.report_prefix}-Failure occurs in Torchinductor Training Benchmark (${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }           
        }//training training_full
    }//email
}
