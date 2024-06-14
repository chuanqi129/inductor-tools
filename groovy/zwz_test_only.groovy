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

node("inductor_gnr_801095") {
    cleanup()
    retry(3){
        checkout([
            $class: 'GitSCM',
            branches: scm.branches,
            doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
            extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
            userRemoteConfigs: scm.userRemoteConfigs
        ])
        
        sleep(60)
    }
    
}