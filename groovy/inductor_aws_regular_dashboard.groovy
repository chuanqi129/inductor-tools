NODE_LABEL = 'yudongsi-mlt-ace'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"


debug = 'True'
if ('debug' in params) {
    echo "debug in params"
    if (params.debug != '') {
        debug = params.debug
    }
}
echo "debug: $debug"

aws_hostname = 'ec2-34-207-213-120.compute-1.amazonaws.com'
if ('aws_hostname' in params) {
    echo "aws_hostname in params"
    if (params.aws_hostname != '') {
        aws_hostname = params.aws_hostname
    }
}
echo "aws_hostname: $aws_hostname"


target = 'WW17.2'
if ('target' in params) {
    echo "target in params"
    if (params.target != '') {
        target = params.target
    }
}
echo "target: $target"


reference = 'WW16.4'
if ('reference' in params) {
    echo "reference in params"
    if (params.reference != '') {
        reference = params.reference
    }
}
echo "reference: $reference"


env._name = "$aws_hostname"
env._target = "$target"
env._reference = "$reference"
env._archive_dir = '/home2/yudongsi/inductor_dashboard'

env._time = new Date().format('yyyy-MM-dd')
println(env._time)

node(NODE_LABEL){
    stage("login instance && launch benchmark") {
        deleteDir()
        retry(3){
            sh '''
            #!/usr/bin/env bash
            cd /home2/yudongsi/
            cat .ssh/config
            sed -i -e "/Host icx-yudong-new/{n;s/.*/    Hostname ${_name}/}" .ssh/config
            cat .ssh/config
            ssh ubuntu@icx-yudong-new "nohup bash entrance.sh ${_target} &>/dev/null &" &
            '''
        }
    }
    stage("acquire logs"){
        retry(3){
            sh '''
            #!/usr/bin/env bash
            set +e
            for ((i=1;i<=25;i++))
            do
                ssh ubuntu@icx-yudong-new "test -f /home/ubuntu/docker/finished.txt"
                if [ $? -eq 0 ]; then
                    if [ -d ${_archive_dir}/${_target} ]; then
                        rm -rf ${_archive_dir}/${_target}
                    fi
                    mkdir -p ${_archive_dir}/${_target}
                    scp -r ubuntu@icx-yudong-new:/home/ubuntu/docker/inductor_log ${_archive_dir}/${_target}
                    break
                else
                    sleep 1h
                    echo $t
                fi
            done
            '''
        }
    }

    stage("generate report"){
        retry(3){
            sh '''
            #!/usr/bin/env bash          
            cd ${_archive_dir} && python report.py -r ${_reference} -t ${_target} -m all
            '''
        }
    }    

    stage('archiveArtifacts') {
            sh '''
            #!/usr/bin/env bash
            cp -r ${_archive_dir}/${_target}/inductor_log ${WORKSPACE}
            '''        
        archiveArtifacts artifacts: "**/inductor_log/**", fingerprint: true
    }

    stage("Sent Email"){
        if ("${debug}" == "true"){
            maillist="yudong.si@intel.com"
        }else{
            maillist="Chuanqi.Wang@intel.com;guobing.chen@intel.com;beilei.zheng@intel.com;xiaobing.zhang@intel.com;xuan.liao@intel.com;Chunyuan.Wu@intel.com;Haozhe.Zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;eikan.wang@intel.com;fan.zhao@intel.com;shufan.wu@intel.com;weizhuo.zhang@intel.com;yudong.si@intel.com;diwei.sun@intel.com"
        }
        if (fileExists("${WORKSPACE}/inductor_log/inductor_model_bench.html") == true){
            emailext(
                subject: "Torchinductor Regularly Benchmark Report (AWS)",
                mimeType: "text/html",
                attachmentsPattern: "**/inductor_log/*.xlsx",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: '${FILE,path="inductor_log/inductor_model_bench.html"}'
            )
        }else{
            emailext(
                subject: "Failure occurs in Torchinductor Regularly Benchmark (AWS)",
                mimeType: "text/html",
                from: "pytorch_inductor_val@intel.com",
                to: maillist,
                body: 'Job build failed, please double check in ${BUILD_URL}'
            )
        }
    }//email
}