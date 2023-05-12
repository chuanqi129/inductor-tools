NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

DYNAMO_BACKEND = 'ipex'
if ('DYNAMO_BACKEND' in params) {
    echo "DYNAMO_BACKEND in params"
    if (params.NODE_LABEL != '') {
        DYNAMO_BACKEND = params.DYNAMO_BACKEND
    }
}
echo "DYNAMO_BACKEND: $DYNAMO_BACKEND"

def get_time(){
    return new Date().format('yyyy-MM-dd')
}
env._VERSION = get_time()
println(env._VERSION)

env._NODE = "$NODE_LABEL"
env.DOCKER_IMAGE_NAMESPACE = 'ccr-registry.caas.intel.com/pytorch/pytorch-ipex-spr'

node(NODE_LABEL){
    stage("get image and inductor-tools repo clone"){
        echo 'get image and inductor-tools repo clone......'
        if ("${isPull}" == "true") {
            def image_build_job = build job: 'inductor_images', propagate: false, parameters: [
                [$class: 'StringParameterValue', name: 'NODE_LABEL', value: "${IMAGE_BUILD_NODE}"],
                [$class: 'StringParameterValue', name: 'BASE_IMAGE', value: "${BASE_IMAGE}"],                
                [$class: 'StringParameterValue', name: 'PT_REPO', value: "${PT_REPO}"],
                [$class: 'StringParameterValue', name: 'PT_BRANCH', value: "${PT_BRANCH}"],
                [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${PT_COMMIT}"],
                [$class: 'StringParameterValue', name: 'IPEX_REPO', value: "${IPEX_REPO}"],
                [$class: 'StringParameterValue', name: 'IPEX_BRANCH', value: "${IPEX_BRANCH}"],
                [$class: 'StringParameterValue', name: 'IPEX_COMMIT', value: "${IPEX_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_VISION_BRANCH', value: "${TORCH_VISION_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_VISION_COMMIT', value: "${TORCH_VISION_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_TEXT_BRANCH', value: "${TORCH_TEXT_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_TEXT_COMMIT', value: "${TORCH_TEXT_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_DATA_BRANCH', value: "${TORCH_DATA_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_DATA_COMMIT', value: "${TORCH_DATA_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_AUDIO_BRANCH', value: "${TORCH_AUDIO_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_AUDIO_COMMIT', value: "${TORCH_AUDIO_COMMIT}"],
                [$class: 'StringParameterValue', name: 'TORCH_BENCH_BRANCH', value: "${TORCH_BENCH_BRANCH}"],
                [$class: 'StringParameterValue', name: 'TORCH_BENCH_COMMIT', value: "${TORCH_BENCH_COMMIT}"],
                [$class: 'StringParameterValue', name: 'BENCH_COMMIT', value: "${BENCH_COMMIT}"],
                [$class: 'StringParameterValue', name: 'tag', value: "${image_tag}"],
            ]
            task_status = image_build_job.result
            task_number = image_build_job.number
            withEnv(["task_status=${task_status}","task_number=${task_number}"]) {
                sh '''#!/bin/bash
                    tag=${image_tag}
                    old_container=`docker ps |grep pytorch-ipex-spr:nightly |awk '{print $1}'`
                    if [ -n "${old_container}" ]; then
                        docker stop $old_container
                        docker rm $old_container
                        docker container prune -f
                    fi
                    old_image_id=`docker images|grep pytorch-ipex-spr|grep nightly|awk '{print $3}'`
                    old_image=`echo $old_image_id| awk '{print $1}'`
                    if [ -n "${old_image}" ]; then
                        docker rmi -f $old_image
                    fi 
                    cd ${WORKSPACE}
                    rm -rf tmp
                    git clone -b ${inductor_tools_branch} https://github.com/chuanqi129/inductor-tools.git tmp                    
                    if [ ${task_status} == "SUCCESS" ]; then
                        docker system prune -f
                        docker login ccr-registry.caas.intel.com -u yudongsi -p 1996+SYD
                        docker pull ${DOCKER_IMAGE_NAMESPACE}:${tag}
                    fi
                '''
            }        
        }else {
            sh '''
            #!/usr/bin/env bash
            tag=${image_tag}
            old_container=`docker ps |grep pt_inductor:nightly |awk '{print $1}'`
            if [ -n "${old_container}" ]; then
                docker stop $old_container
                docker rm $old_container
                docker container prune -f
            fi
            old_image_id=`docker images|grep pt_inductor|grep nightly|awk '{print $3}'`
            old_image=`echo $old_image_id| awk '{print $1}'`
            if [ -n "${old_image}" ]; then
                docker rmi -f $old_image
            fi
            docker login ccr-registry.caas.intel.com -u yudongsi -p 1996+SYD
            docker pull ${DOCKER_IMAGE_NAMESPACE}:${tag}            
            '''
        } 
    }
    stage('login container & cp scripts & run') {
        echo 'login container & cp scripts & run......'
        if ("${isOP}" == "true") {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name op_pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log/build_num_$BUILD_NUMBER:/workspace/pytorch/dynamo_opbench ${DOCKER_IMAGE_NAMESPACE}:nightly
            docker cp tmp/scripts/microbench/microbench_parser.py op_pt_inductor:/workspace/pytorch
            docker cp tmp/scripts/microbench/microbench.sh op_pt_inductor:/workspace/pytorch
            docker exec -i op_pt_inductor bash -c "bash microbench.sh dynamo_opbench ${op_suite} ${op_repeats};cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py -w ${_VERSION} -l ${BUILD_URL} -n ${_NODE};rm microbench_parser.py"
            '''
        }else {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/inductor_log/build_num_$BUILD_NUMBER:/workspace/pytorch/inductor_log ${DOCKER_IMAGE_NAMESPACE}:nightly
            docker cp tmp/scripts/modelbench/inductor_test.sh pt_inductor:/workspace/pytorch         
            docker cp tmp/scripts/modelbench/log_parser.py pt_inductor:/workspace/pytorch           
            docker exec -i pt_inductor bash inductor_test.sh ${THREAD} ${CHANNELS} inductor_log ${MODEL_SUITE}
            '''
        }
    }
    stage('archiveArtifacts') {
        if ("${isOP}" == "true"){
            archiveArtifacts artifacts: "**/opbench_log/build_num_$BUILD_NUMBER/**", fingerprint: true
        }else{
            archiveArtifacts artifacts: "**/inductor_log/build_num_$BUILD_NUMBER/**", fingerprint: true
        }
    }
    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/opbench_log/build_num_$BUILD_NUMBER/op-microbench-${_VERSION}.xlsx") == true){
            emailext(
                subject: "Torchinductor OP microbench Nightly Report",
                mimeType: "text/html",
                attachmentsPattern: "**/opbench_log/build_num_$BUILD_NUMBER/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="opbench_log/build_num_$BUILD_NUMBER/ops.html"}'
            )
        }
        else{
            emailext(
                subject: "Failure occurs in Torchinductor OP microbench Nightly",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    } 
}