NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

def get_time(){
    return new Date().format('yyyy-MM-dd')
}
env._VERSION = get_time()
println(env._VERSION)

env._NODE = "$NODE_LABEL"


node(NODE_LABEL){
    stage("get image and inductor-tools repo clone"){
        echo 'get image and inductor-tools repo clone......'
        if ("${isPull}" == "true") {
            sh '''
            #!/usr/bin/env bash
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
            cd ${WORKSPACE}
            rm -rf tmp
            git clone -b ${inductor_tools_branch} https://github.com/chuanqi129/inductor-tools.git tmp
            docker login ccr-registry.caas.intel.com
            docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            '''            
        }else {
            sh '''
            #!/usr/bin/env bash
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
            cd ${WORKSPACE}
            rm -rf tmp
            git clone -b ${inductor_tools_branch} https://github.com/chuanqi129/inductor-tools.git tmp
            mv tmp/docker/Dockerfile ./
            DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg PT_REPO=${PT_REPO} --build-arg PT_BRANCH=${PT_BRANCH} --build-arg PT_COMMIT=${PT_COMMIT} --build-arg TORCH_VISION_BRANCH=${TORCH_VISION_BRANCH} --build-arg TORCH_VISION_COMMIT=${TORCH_VISION_COMMIT} --build-arg TORCH_DATA_BRANCH=${TORCH_DATA_BRANCH} --build-arg TORCH_DATA_COMMIT=${TORCH_DATA_COMMIT} --build-arg TORCH_TEXT_BRANCH=${TORCH_TEXT_BRANCH} --build-arg TORCH_TEXT_COMMIT=${TORCH_TEXT_COMMIT} --build-arg TORCH_AUDIO_BRANCH=${TORCH_AUDIO_BRANCH} --build-arg TORCH_AUDIO_COMMIT=${TORCH_AUDIO_COMMIT} --build-arg TORCH_BENCH_BRANCH=${TORCH_BENCH_BRANCH} --build-arg TORCH_BENCH_COMMIT=${TORCH_BENCH_COMMIT} --build-arg BENCH_COMMIT=${BENCH_COMMIT} -t ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly -f Dockerfile --target image .
            '''
        } 
    }
    stage('login container & cp scripts & run') {
        echo 'login container & cp scripts & run......'

        if ("${isOP}" == "true") {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name op_pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log/build_num_$BUILD_NUMBER:/workspace/pytorch/dynamo_opbench ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker cp tmp/scripts/microbench/microbench_parser.py op_pt_inductor:/workspace/pytorch
            docker cp tmp/scripts/microbench/microbench.sh op_pt_inductor:/workspace/pytorch
            docker exec -i op_pt_inductor bash -c "bash microbench.sh dynamo_opbench ${op_suite} ${op_repeats};cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py -w ${_VERSION} -l ${BUILD_URL} -n ${_NODE};rm microbench_parser.py"
            '''
        }else {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/inductor_log/build_num_$BUILD_NUMBER:/workspace/pytorch/inductor_log ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
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
        maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        maillist_debug="yudong.si@intel.com"
        if (fileExists("${WORKSPACE}/opbench_log/build_num_$BUILD_NUMBER/op-microbench-${_VERSION}.xlsx") == true){
            emailext(
                subject: "Torchinductor OP microbench Nightly Report",
                mimeType: "text/html",
                attachmentsPattern: "**/opbench_log/build_num_$BUILD_NUMBER/*.xlsx",
                from: "Inductor_op_microbench_nightly@intel.com",
                to: maillist,
                body: '${FILE,path="opbench_log/build_num_$BUILD_NUMBER/ops.html"}'
            )
        }
        else{
            emailext(
                subject: "Failure occurs in Torchinductor OP microbench Nightly",
                mimeType: "text/html",
                from: "Inductor_op_microbench_nightly@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    } 
}