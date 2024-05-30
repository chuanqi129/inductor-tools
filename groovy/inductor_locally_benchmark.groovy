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
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pt_inductor'
env.BASE_IMAGE= 'ubuntu:22.04'

def cleanup(){
    try {
        sh'''
            #!/usr/bin/env bash
            docker stop $(docker ps -a -q)
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

node(NODE_LABEL){
    stage("prepare"){
        println('prepare......')
        // TODO: implement report_only logic
        if  ("${report_only}" == "false") {
            cleanup()
            pruneOldImage()
            checkout scm
            sh'''
                #!/usr/bin/env bash
                mkdir -p ${WORKSPACE}/${target}
            '''
        }
    }

    stage("prepare container"){
        if  ("${report_only}" == "false") {
            if (TORCH_COMMIT == "nightly") {
                withCredentials([usernamePassword(credentialsId: 'caas_docker_hub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
                    withCredentials([usernamePassword(credentialsId: 'syd_token_inteltf-jenk', usernameVariable: 'TG_USERNAME', passwordVariable: 'TG_PASSWORD')]){
                    sh'''
                        #!/usr/bin/env bash
                        # clone pytorch repo
                        cd ${WORKSPACE}
                        git clone -b ${TORCH_COMMIT} --depth 1 ${TORCH_REPO}
                        cd pytorch
                        commit_date=`git log --format="%cs"`
                        bref_commit=`git log --pretty=format:"%s" -1 | cut -d '(' -f2 | cut -d ')' -f1 | cut -c 1-7`
                        DOCKER_TAG="${commit_date}-${bref_commit}"

                        docker login ccr-registry.caas.intel.com -u $USERNAME -p $PASSWORD
                        docker manifest inspect ${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG}
                        status=${PIPESTATUS[0]}
                        if [ "$status" != "0" ];then
                            # build docker image, because the target images does not exist
                            DOCKER_BUILDKIT=1 docker build \
                                --no-cache \
                                --build-arg http_proxy=${http_proxy} \
                                --build-arg https_proxy=${https_proxy} \
                                --build-arg PT_REPO=$TORCH_REPO \
                                --build-arg PT_COMMIT=$TORCH_COMMIT \
                                --build-arg BENCH_COMMIT=$DYNAMO_BENCH \
                                --build-arg TORCH_AUDIO_COMMIT=$AUDIO \
                                --build-arg TORCH_TEXT_COMMIT=$TEXT \
                                --build-arg TORCH_VISION_COMMIT=$VISION \
                                --build-arg TORCH_DATA_COMMIT=$DATA \
                                --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH \
                                --build-arg HF_HUB_TOKEN=$HF_TOKEN \
                                -t ${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG} \
                                -f docker/Dockerfile --target image . > ${target}/docker_image_build.log 2>&1
                            docker_build_result=${PIPESTATUS[0]}
                            # Early exit for docker image build issue
                            if [ "$docker_build_result" != "0" ];then
                                echo "Docker image build failed, early exit!"
                                exit 1
                            fi
                            docker push ${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG}
                        else
                            echo "[INFO] pull existed image ${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG}"
                            docker pull ${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG} > ${target}/docker_image_build.log 2>&1
                        fi
                        echo "${DOCKER_IMAGE_NAMESPACE}:${DOCKER_TAG}" > ${target}/docker_image_name.log
                    '''
                    }
                }    
            } else {
                sh '''
                    #!/usr/bin/env bash
                    # build docker image
                    DOCKER_BUILDKIT=1 docker build \
                        --no-cache \
                        --build-arg http_proxy=${http_proxy} \
                        --build-arg https_proxy=${https_proxy} \
                        --build-arg PT_REPO=$TORCH_REPO \
                        --build-arg PT_COMMIT=$TORCH_COMMIT \
                        --build-arg BENCH_COMMIT=$DYNAMO_BENCH \
                        --build-arg TORCH_AUDIO_COMMIT=$AUDIO \
                        --build-arg TORCH_TEXT_COMMIT=$TEXT \
                        --build-arg TORCH_VISION_COMMIT=$VISION \
                        --build-arg TORCH_DATA_COMMIT=$DATA \
                        --build-arg TORCH_BENCH_COMMIT=$TORCH_BENCH \
                        --build-arg HF_HUB_TOKEN=$HF_TOKEN \
                        -t pt_inductor_tmp:${target} \
                        -f docker/Dockerfile --target image . > ${target}/docker_image_build.log 2>&1
                    docker_build_result=${PIPESTATUS[0]}
                    # Early exit for docker image build issue
                    if [ "$docker_build_result" != "0" ];then
                        echo "Docker image build failed, early exit!"
                        exit 1
                    fi
                    echo "pt_inductor_tmp:${target}" > ${target}/docker_image_name.log
                '''
            }
        }
        if (fileExists("${WORKSPACE}/${target}/docker_image_build.log")) {
            archiveArtifacts  "${target}/docker_image*"
        }
    }

    stage("benchmark") {
        if  ("${report_only}" == "false") {
            sh '''
                #!/usr/bin/env bash
                docker_image_name=`cat ${target}/docker_image_name.log`
                docker run -tid --name $USER \
                    --privileged \
                    --env https_proxy=${https_proxy} \
                    --env http_proxy=${http_proxy} \
                    --net host --shm-size 1G \
                    -v ~/.cache:/root/.cache \
                    -v ${WORKSPACE}/${target}:/workspace/pytorch/${target} \
                    ${docker_image_name}
                docker cp scripts/modelbench/inductor_test.sh $USER:/workspace/pytorch
                docker cp scripts/modelbench/inductor_train.sh $USER:/workspace/pytorch
                docker cp scripts/modelbench/version_collect.sh $USER:/workspace/pytorch
                docker exec -i $USER bash -c "bash version_collect.sh ${target} $DYNAMO_BENCH"

                if [ $test_mode == "inference" ]; then
                    docker exec -i $USER bash -c "bash inductor_test.sh $THREADS $CHANNELS $precision $shape $target $WRAPPER $HF_TOKEN $backend inference $suite $extra_param"
                elif [ $test_mode == "training_full" ]; then
                    docker exec -i $USER bash -c "bash inductor_test.sh multiple $CHANNELS $precision $shape $target $WRAPPER $HF_TOKEN $backend training $suite $extra_param"
                elif [ $test_mode == "training" ]; then
                    docker exec -i $USER bash -c "bash inductor_train.sh $CHANNELS $precision $target $extra_param"
                fi
            '''
        }
    }

    // Add raw log artifact stage in advance to avoid crash in report generate stage
    stage("archive raw test results"){
        sh '''
            #!/usr/bin/env bash
            cp -r  ${WORKSPACE}/${target} ${WORKSPACE}/raw_log
        '''
        archiveArtifacts artifacts: "**/raw_log/**", fingerprint: true
    }

    stage("stop docker") {
        sh'''
            #!/usr/bin/env bash
            docker stop $(docker ps -a -q)
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
                    cd ${WORKSPACE}
                    cp scripts/modelbench/report.py ${WORKSPACE}
                    python report.py \
                        -r refer \
                        -t ${target} \
                        -m ${THREADS} \
                        --precision ${precision} \
                        --url ${BUILD_URL} \
                        --image_tag ${target}_aws \
                        --suite ${suite} \
                        --infer_or_train ${infer_or_train} \
                        --shape ${shape} \
                        --wrapper ${WRAPPER} \
                        --torch_repo ${TORCH_REPO} \
                        --backend ${backend} \
                        ${dashboard_args}
                    rm -rf refer
                '''
            }else{
                sh '''
                    #!/usr/bin/env bash
                    cd ${WORKSPACE}
                    cp scripts/modelbench/report.py ${WORKSPACE}
                    python report.py \
                        -t ${target} \
                        -m ${THREADS} \
                        --precision ${precision} \
                        --url ${BUILD_URL} \
                        --image_tag ${target}_aws \
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
                cp -r inductor_log refer
                rm -rf inductor_log
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
            rm -rf ${WORKSPACE}/raw_log
        '''
        if ("${test_mode}" == "inference" || "${test_mode}" == "training_full")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard
            cp -r  ${WORKSPACE}/${target} $HOME/inductor_dashboard
            cd ${WORKSPACE} && mv ${WORKSPACE}/${target}/inductor_log/ ./ && rm -rf ${target}
            '''
        }
        if ("${test_mode}" == "training")
        {
            sh '''
            #!/usr/bin/env bash
            mkdir -p $HOME/inductor_dashboard/Train
            cp -r  ${WORKSPACE}/${target} $HOME/inductor_dashboard/Train
            cd ${WORKSPACE} && mv ${WORKSPACE}/${target}/inductor_log/ ./ && rm -rf ${target}
            '''
        } 
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
        if (fileExists("${WORKSPACE}/guilty_commit_search_model_list.csv")) {
            archiveArtifacts  "guilty_commit_search*"
        }
        if (fileExists("${WORKSPACE}/all_model_list.csv")) {
            archiveArtifacts  "all_model_list.csv"
        }
    }

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="${debug_mail}"
        }else{
            maillist="${default_mail}"
        }
        if ("${test_mode}" == "inference")
        {
            if (fileExists("${WORKSPACE}/inductor_log/inductor_model_bench.html") == true){
                emailext(
                    subject: "Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-Report(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }
        }//inference
        if ("${test_mode}" == "training" || "${test_mode}" == "training_full")
        {
            if (fileExists("${WORKSPACE}/inductor_log/inductor_model_training_bench.html") == true){
                emailext(
                    subject: "Torchinductor-${env.backend}-${env.test_mode}-${env.precision}-${env.shape}-${env.WRAPPER}-Report(${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    attachmentsPattern: "**/inductor_log/*.xlsx",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: '${FILE,path="inductor_log/inductor_model_training_bench.html"}'
                )
            }else{
                emailext(
                    subject: "Failure occurs in Torchinductor Training Benchmark (${env.bench_machine})_${env.target}",
                    mimeType: "text/html",
                    from: "pytorch_inductor_val@intel.com",
                    to: maillist,
                    body: 'Job build failed, please double check in ${BUILD_URL}'
                )
            }           
        }//training training_full
    }//email
}
