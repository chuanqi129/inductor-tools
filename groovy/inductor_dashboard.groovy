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

def AdditionalInfo() {
    return """<p>Torchinductor OP microbench Nightly Report
                  <p>job info:</p>
                  <ol>
                    <table>
                      <tbody>
                        <tr>
                          <td>Build url:&nbsp;</td>
                          <td>${BUILD_URL}</td>
                        </tr>
                    </table>       
                  </ol>     
                  <p>SW Info:</p>
                   <ol>
                      <table>
                        <tbody>
                          <tr>
                            <td>docker image:&nbsp;</td>
                            <td>ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly</td>
                          </tr>
                          <tr>
                            <td>StockPT:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/pytorch/commits/nightly">${_VERSION} nightly release</a></td>
                          </tr>
                          <tr>
                            <td>TORCH_VISION:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/vision/commits/nightly/">${_VERSION} nightly release</a></td>
                          </tr>
                          <tr>
                            <td>TORCH_TEXT:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/text/commits/nightly/">${_VERSION} nightly release</a></td>
                          </tr>
                          <tr>
                            <td>TORCH_AUDIO:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/audio/commits/nightly/">${_VERSION} nightly release</a></td>
                          </tr>
                          <tr>
                            <td>TORCH_DATA:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/data/commits/nightly/">${_VERSION} nightly release</a></td>
                          </tr>
                          <tr>
                            <td>TORCH_BENCH:&nbsp;</td>
                            <td><a href="https://github.com/pytorch/benchmark/commits/main">${_VERSION} torchbench</a></td>
                          </tr>
                        </tbody>
                        <colgroup>
                          <col>
                          <col>
                        </colgroup>
                      </table>    
                   </ol>
                  <p>HW info:</p>
                    <ol>
                    <table>
                      <tbody>
                        <tr>
                          <td>Machine name:&nbsp;</td>
                          <td>$NODE_LABEL</td>
                        </tr>                  
                        <tr>
                          <td>Manufacturer:&nbsp;</td>
                          <td>Intel Corporation</td>
                        </tr>
                        <tr>
                          <td>Kernel:</td>
                          <td>5.4.0-131-genericL</td>
                        </tr>
                        <tr>
                          <td>Microcode:</td>
                          <td>0xd000375</td>
                        </tr>
                        <tr>
                          <td>Installed Memory:</td>
                          <td>503GB</td>
                        </tr>
                        <tr>
                          <td>OS:</td>
                          <td>Ubuntu 18.04.6 LTS</td>
                        </tr>
                        <tr>
                          <td>CPU Model:</td>
                          <td>Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz</td>
                        </tr>
                        <tr>
                          <td>GCC:</td>
                          <td>gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0</td>
                        </tr>
                        <tr>
                          <td>GLIBC:</td>
                          <td>ldd (Ubuntu GLIBC 2.27-3ubuntu1.5) 2.27</td>
                        </tr>
                        <tr>
                          <td>Binutils:</td>
                          <td>GNU ld (GNU Binutils for Ubuntu) 2.30</td>
                        </tr>
                        <tr>
                          <td>Python:</td>
                          <td>Python 3.8.3</td>
                        </tr>
                      </tbody>
                      <colgroup>
                        <col>
                        <col>
                      </colgroup>
                    </table>
                    </ol> 
                  <p>Todo: regression check comparison [Dashboard]</p>
                  <p>You can find details from attachments, Thanks</p>
              </p>
           """
}    

node(NODE_LABEL){
    stage("get image"){
        echo 'get image......'
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
            docker login ccr-registry.caas.intel.com
            docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            '''            
        }else {
            sh '''
            #!/usr/bin/env bash
            cd ${WORKSPACE}
            rm -rf tmp
            git clone -b ${inductor_tools_branch} https://github.com/chuanqi129/inductor-tools.git tmp
            mv tmp/scripts/microbench/microbench_parser.py ./
            mv tmp/scripts/microbench/microbench.sh ./
            mv tmp/scripts/modelbench/inductor_test.sh ./         
            mv tmp/scripts/modelbench/log_parser.py ./
            DOCKER_BUILDKIT=1 docker build --no-cache --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg PT_REPO=${PT_REPO} --build-arg PT_BRANCH=${PT_BRANCH} --build-arg PT_COMMIT=${PT_COMMIT} --build-arg TORCH_VISION_BRANCH=${TORCH_VISION_BRANCH} --build-arg TORCH_VISION_COMMIT=${TORCH_VISION_COMMIT} --build-arg TORCH_DATA_BRANCH=${TORCH_DATA_BRANCH} --build-arg TORCH_DATA_COMMIT=${TORCH_DATA_COMMIT} --build-arg TORCH_TEXT_BRANCH=${TORCH_TEXT_BRANCH} --build-arg TORCH_TEXT_COMMIT=${TORCH_TEXT_COMMIT} --build-arg TORCH_AUDIO_BRANCH=${TORCH_AUDIO_BRANCH} --build-arg TORCH_AUDIO_COMMIT=${TORCH_AUDIO_COMMIT} --build-arg TORCH_BENCH_BRANCH=${TORCH_BENCH_BRANCH} --build-arg TORCH_BENCH_COMMIT=${TORCH_BENCH_COMMIT} --build-arg BENCH_COMMIT=${BENCH_COMMIT} -t ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly -f Dockerfile --target image .
            '''
        } 
    }
    stage('login container and test') {
        echo 'login container and test......'
        if ("${isOP}" == "true") {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name op_pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/opbench_log/build_num_$BUILD_NUMBER:/workspace/pytorch/dynamo_opbench ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly
            docker exec -i op_pt_inductor bash -c "bash microbench.sh dynamo_opbench ${op_suite} ${op_repeats};cp microbench_parser.py dynamo_opbench;cd dynamo_opbench;pip install openpyxl;python microbench_parser.py --workday ${_VERSION};rm microbench_parser.py"
            '''
        }else {
            sh '''
            #!/usr/bin/env bash
            docker run -tid --name pt_inductor --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host  --shm-size 1G -v ${WORKSPACE}/inductor_log/build_num_$BUILD_NUMBER:/workspace/pytorch/inductor_log ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly        
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
        emailext(
            subject: "Torchinductor OP microbench Nightly Report",
            mimeType: "text/html",
            attachmentsPattern: "**/opbench_log/build_num_$BUILD_NUMBER/*.xlsx",
            from: "Inductor_op_microbench_nightly@intel.com",
            to: maillist,
            body: AdditionalInfo()
        )
    }    
}
