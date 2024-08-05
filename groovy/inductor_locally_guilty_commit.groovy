env.bench_machine = "Local"
env.target = new Date().format('yyyy_MM_dd')
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE= 'ubuntu:22.04'
env.LOG_DIR = 'inductor_log'
if (env.NODE_LABEL == "0") {
    if (env.precision == "float32") {
        env.NODE_LABEL = "inductor-icx-local"
    } else if (env.precision == 'amp') {
        env.NODE_LABEL = "inductor-gnr-local"
    }
}

def cleanup(){
    try {
        sh'''
            #!/usr/bin/env bash
            docker_ps=`docker ps -a -q`
            if [ -n "${docker_ps}" ];then
                docker stop ${docker_ps}
            fi
            docker container prune -f
            docker system prune -f
            docker pull ${BASE_IMAGE}
        '''
        docker.image(env.BASE_IMAGE).inside(" \
            -u root \
            -v ${WORKSPACE}:/root/workspace \
            --privileged \
        "){
        sh '''
            chmod -R 777 /root/workspace    
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

node(us_node) {
    stage("prepare torch repo"){
        deleteDir()
        sh'''
            #!/usr/bin/env bash
            cd ${WORKSPACE}
            git clone ${TORCH_REPO}
            cd pytorch
            git checkout ${TORCH_END_COMMIT}
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
}

node(NODE_LABEL){
    stage("prepare"){
        println('prepare......')
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
            mkdir -p ${WORKSPACE}/${target}/${LOG_DIR}
            mv docker_image_tag.log ${WORKSPACE}/${target}/${LOG_DIR}
        '''
    }

    stage("trigger inductor images job"){
        def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/${target}/${LOG_DIR}/docker_image_tag.log''').toString().trim().replaceAll("\n","")
        def image_build_job = build job: 'inductor_images_local', propagate: false, parameters: [             
            [$class: 'StringParameterValue', name: 'PT_REPO', value: "${TORCH_REPO}"],
            [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${TORCH_START_COMMIT}"],
            [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
            [$class: 'StringParameterValue', name: 'HF_TOKEN', value: "${HF_TOKEN}"],
        ]
        if (fileExists("${WORKSPACE}/${target}/${LOG_DIR}/docker_image_tag.log")) {
            archiveArtifacts  "${target}/${LOG_DIR}/docker_image_tag.log"
        }
    }

    stage("prepare container"){
        withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
            sh'''
                #!/usr/bin/env bash
                docker_image_tag=`cat ${target}/${LOG_DIR}/docker_image_tag.log`
                docker pull ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
            '''
        }
    }

    stage("benchmark") {
        sh '''
            #!/usr/bin/env bash
            docker_image_tag=`cat ${target}/${LOG_DIR}/docker_image_tag.log`
            docker run -tid --name $USER \
                --privileged \
                --env https_proxy=${https_proxy} \
                --env http_proxy=${http_proxy} \
                --env HF_HUB_TOKEN=$HF_TOKEN \
                --net host --shm-size 1G \
                -v ~/.cache:/root/.cache \
                -v ${WORKSPACE}/${target}/${LOG_DIR}:/workspace/pytorch/${LOG_DIR} \
                ${DOCKER_IMAGE_NAMESPACE}:${docker_image_tag}
            docker cp scripts/modelbench/bisect_search.sh $USER:/workspace/pytorch
            docker cp scripts/modelbench/bisect_run_test.sh $USER:/workspace/pytorch
            docker cp scripts/modelbench/inductor_single_run.sh $USER:/workspace/pytorch
            # TODO: Hard code freeze on and default bs, add them as params future
            docker exec -i $USER bash -c "bash bisect_search.sh \
                START_COMMIT=$TORCH_START_COMMIT \
                END_COMMIT=$TORCH_END_COMMIT \
                SUITE=$suite \
                MODEL=$model \
                MODE=$test_mode \
                SCENARIO=$scenario \
                PRECISION=$precision \
                SHAPE=$shape \
                WRAPPER=$WRAPPER \
                KIND=$kind \
                THREADS=$THREADS \
                CHANNELS=$CHANNELS \
                BS=0 \
                LOG_DIR=$LOG_DIR \
                HF_TOKEN=$HF_TOKEN \
                BACKEND=$backend \
                PERF_RATIO=$perf_ratio \
                EXTRA=$extra_param" \
                > ${WORKSPACE}/${target}/${LOG_DIR}/docker_exec_detailed.log

            docker exec -i $USER bash -c "chmod 777 -R /workspace/pytorch/${LOG_DIR}"
        '''
    }

    stage('archiveArtifacts') {
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
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

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="${default_mail}"
        }

        if (fileExists("${WORKSPACE}/inductor_log/guilty_commit.log") == true){
            emailext(
                subject: "Torchinductor-${env.backend}-${env.suite}-${env.model}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-${env.threads}-${env.scenario}-${env.kind}-guilty_commit_Report(${env.bench_machine})_${env.target}",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*_guilty_commit.log",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build succeed, please double check in ${BUILD_URL}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Torchinductor-${env.backend}-${env.suite}-${env.model}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-${env.threads}-${env.scenario}-${env.kind}-guilty_commit_Report(${env.bench_machine})_${env.target}",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}
