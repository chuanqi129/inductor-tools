NODE_LABEL = 'mlp-validate-icx24-ubuntu'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

def cleanup(){
    sh '''
        #!/usr/bin/env bash
        cd ${WORKSPACE}
        sudo rm -rf *
    '''
}

node(NODE_LABEL){
    stage("Clean"){
        cleanup()
        deleteDir()
    }

    stage("Trigger") {
        withCredentials([usernamePassword(credentialsId: 'remote_trigger_token', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]){
            sh '''
            #!/usr/bin/env bash
            curl -s -I -k -u $USERNAME:$PASSWORD "https://jenkins-aten-caffe2.sh.intel.com/job/inductor_locally_regular/buildWithParameters?token=SYD_TOKEN"
            if [ $? -eq 0 ]; then
                 echo "trigger successfully"
            else
                 echo "trigger failed"
            fi
            '''              
        }
    }     
}