import hudson.model.Computer
import hudson.model.Label

def str_list = ['target', 'baseline']
def maxRetries = 3
env.benchmark_job = 'inductor_locally_benchmark'
env.result_compare_job = 'inductor_job_result_compare'
env.target_job_selector = 'None'
env.baseline_job_selector = 'None'
if (env.precision == "float32") {
    env.labelName = "inductor-icx-local-tas"
} else if (env.precision == 'amp-sh') {
    env.precision == 'amp'
    env.labelName = "inductor-gnr-local-tas-sh"
} else if (env.precision == 'amp-us') {
    env.precision == 'amp'
    env.labelName = "inductor-gnr-local-tas-us"
}

def getAvailableNode(String labelName) {
    Label label = Jenkins.instance.getLabel(labelName)
    
    // Get computers by label
    Collection<Computer> computers = label.getNodes().collect { it.toComputer() }
    
    String availableComputer = null
    for (Computer computer : computers) {
      if (computer.isOnline() && computer.isIdle()) {
        availableComputer = computer.name
        break
      }
    }
    return availableComputer
}

def getJobParameters(String test_str, String availableComputer) {
    def job_parameters = [
        string(name: 'TORCH_REPO', value: target_TORCH_REPO),
        string(name: 'TORCH_COMMIT', value: target_TORCH_COMMIT),
        string(name: 'default_mail', value: default_mail),
        string(name: 'backend', value: backend),
        string(name: 'precision', value: precision),
        string(name: 'test_mode', value: test_mode),
        string(name: 'suite', value: suite),
        string(name: 'shape', value: shape),
        string(name: 'THREADS', value: THREADS),
        string(name: 'CHANNELS', value: CHANNELS),
        string(name: 'WRAPPER', value: WRAPPER),
        string(name: 'HF_TOKEN', value: HF_TOKEN),
        string(name: 'extra_param', value: extra_param),
        string(name: 'NODE_LABEL', value: availableComputer),
    ]
    if (test_str == "baseline") {
        println("[INFO]: baseline pytorch repo and commit: ")
        job_parameters[0] = string(name: 'TORCH_REPO', value: baseline_TORCH_REPO)
        job_parameters[1] = string(name: 'TORCH_COMMIT', value: baseline_TORCH_COMMIT)
    }
    return job_parameters
}

node("inductor_image"){
    deleteDir()
    retry(maxRetries){
        sleep(60)
        checkout([
            $class: 'GitSCM',
            branches: scm.branches,
            doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
            extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
            userRemoteConfigs: scm.userRemoteConfigs
        ])
    }

    try{
        stage("Pre-check repo and commit") {
            if ((target_TORCH_REPO == baseline_TORCH_REPO) && 
                (target_TORCH_COMMIT == baseline_TORCH_COMMIT)) {
                println("same repo and commit")
            }
            sh'''
                #!/usr/bin/env bash
                set -ex

                echo "Build URL: ${BUILD_URL}.<br>" | tee ${WORKSPACE}/torch_clone.log
                cd ${WORKSPACE}
                git clone ${target_TORCH_REPO} target_pytorch
                cd ${WORKSPACE}/target_pytorch
                git checkout ${target_TORCH_COMMIT} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                result=${PIPESTATUS[0]}
                if [ "${result}" = "0" ]; then
                    echo "<br>[INFO] Target torch repo and commit is correct.<br>" | tee -a ${WORKSPACE}/torch_clone.log
                else
                    echo "<br>[ERROR] Target torch repo and commit is wrong!<br>" | tee -a ${WORKSPACE}/torch_clone.log
                    exit 1
                fi

                cd ${WORKSPACE}
                git clone ${baseline_TORCH_REPO} baseline_pytorch
                cd ${WORKSPACE}/baseline_pytorch
                git checkout ${baseline_TORCH_COMMIT} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                result=${PIPESTATUS[0]}
                if [ "${result}" = "0" ]; then
                    echo "<br>[INFO] Baseline torch repo and commit is correct.<br>" | tee -a ${WORKSPACE}/torch_clone.log
                else
                    echo "<br>[ERROR] Baseline torch repo and commit is wrong!<br>" | tee -a ${WORKSPACE}/torch_clone.log
                    exit 1
                fi
            '''
        }
    } catch (Exception e) {
        emailext(
            subject: "Inductor TAS pipeline Pre-Check failed",
            mimeType: "text/html",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail,
            body: '${FILE, path="torch_clone.log"}'
        )
        throw e
    }

    try{
        stage("Build target images") {
            retry(maxRetries){
                sh'''
                    #!/usr/bin/env bash
                    set -ex

                    cd ${WORKSPACE}/target_pytorch
                    commit_date=`git log -n 1 --format="%cs"`
                    bref_commit=`git rev-parse --short HEAD`
                    DOCKER_TAG="${commit_date}_${bref_commit}"
                    echo "${DOCKER_TAG}" > ${WORKSPACE}/docker_image_tag_target.log
                '''
                sleep(60)
                def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/docker_image_tag_target.log''').toString().trim().replaceAll("\n","")
                def image_build_job = build job: 'inductor_images_local', propagate: false, parameters: [             
                        [$class: 'StringParameterValue', name: 'PT_REPO', value: "${target_TORCH_REPO}"],
                        [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${target_TORCH_COMMIT}"],
                        [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
                ]

                buildStatus = image_build_job.getResult()
                if (buildStatus == hudson.model.Result.FAILURE) {
                    throw new Exception("Target docker image build job failed")
                }
            }
        }
    } catch (Exception e) {
        sh'''
            #!/usr/bin/env bash
            set -ex
        '''
        emailext(
            subject: "Inductor TAS pipeline Pre-Check failed",
            mimeType: "text/html",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail,
            body: '${FILE, path="torch_clone.log"}'
        )
        throw e
    }
}

node(report_node){
    deleteDir()
    retry(maxRetries){
        sleep(60)
        checkout([
            $class: 'GitSCM',
            branches: scm.branches,
            doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
            extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
            userRemoteConfigs: scm.userRemoteConfigs
        ])
    }

    stage("Pre-check repo and commit") {
        if ((target_TORCH_REPO == baseline_TORCH_REPO) && 
            (target_TORCH_COMMIT == baseline_TORCH_COMMIT)) {
            println("same repo and commit")
        }
    }

    stage('Trigger aws benchmark Job'){
        def job_list = [:]
        def availableComputer = null
        while(true) {
            availableComputer = getAvailableNode(labelName)
            if (availableComputer != null) {
                println("Found available node with label '${labelName}': ${availableComputer}")
                break
            } else {
                println("No available nodes found with label '${labelName}'")
                sleep(600)
            }
        }

        for (sub_str in str_list) {
            def test_str = sub_str
            job_list[test_str] = {
                def job_parameters = getJobParameters(test_str, availableComputer)
                def benchmark_job = build propagate: false,
                    job: benchmark_job, parameters: job_parameters
                
                if (test_str == "target") {
                    env.target_job_selector = benchmark_job.getNumber()
                } else {
                    env.baseline_job_selector = benchmark_job.getNumber()
                }
            } // job_list
        } // for
        parallel job_list
    } // stage 

    // TODO
    // get target and baseline results, and add rebuild button
    // Create rebuild account.
    
    stage('Compare results') {
        if ((target_job_selector == 'None') || 
            (baseline_job_selector == 'None')) {
            println("Target job selector: " + target_job_selector)
            println("Baseline job selector: " + baseline_job_selector)
        }
        def job_parameters = [
            string(name: 'debug_mail', value: default_mail),
            string(name: 'target_job', value: benchmark_job),
            string(name: 'target_job_selector', value: target_job_selector),
            string(name: 'refer_job', value: benchmark_job),
            string(name: 'refer_job_selector', value: baseline_job_selector),
        ]
        def benchmark_job = build propagate: true,
            job: result_compare_job, parameters: job_parameters
    } // stage
}