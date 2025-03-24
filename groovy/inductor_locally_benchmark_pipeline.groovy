import hudson.model.Computer
import hudson.model.Label

def str_list = ['target', 'baseline']
def maxRetries = 3
env.benchmark_job = 'inductor_locally_benchmark'
env.result_compare_job = 'inductor_job_result_compare'
env.target_job_selector = 'None'
env.baseline_job_selector = 'None'
if (env.precision == 'float32') {
    env.labelName = "inductor-icx-local-tas"
} else if (env.precision == 'amp-gnr-sh') {
    env.precision = 'amp'
    env.labelName = "inductor-gnr-local-tas-sh"
} else if (env.precision == 'amp-gnr-us') {
    env.precision = 'amp'
    env.labelName = "inductor-gnr-local-tas-us"
} else if (env.precision == 'amp-fp16-gnr-sh') {
    env.precision = 'amp_fp16'
    env.labelName = "inductor-gnr-local-tas-sh"
} else if (env.precision == 'amp-fp16-gnr-us') {
    env.precision = 'amp_fp16'
    env.labelName = "inductor-gnr-local-tas-us"
} else if (env.precision == 'amp-spr') {
    env.precision = 'amp'
    env.labelName = "inductor-spr-local-tas"
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
        string(name: 'backend', value: target_backend),
        string(name: 'precision', value: precision),
        string(name: 'test_mode', value: test_mode),
        string(name: 'suite', value: suite),
        string(name: 'shape', value: shape),
        string(name: 'THREADS', value: THREADS),
        string(name: 'CHANNELS', value: CHANNELS),
        string(name: 'WRAPPER', value: WRAPPER),
        string(name: 'test_ENV', value: test_ENV),
        string(name: 'HF_TOKEN', value: HF_TOKEN),
        string(name: 'extra_param', value: extra_param),
        string(name: 'NODE_LABEL', value: availableComputer),
    ]
    if (test_str == "baseline") {
        println("[INFO]: baseline pytorch repo and commit: ")
        job_parameters[0] = string(name: 'TORCH_REPO', value: baseline_TORCH_REPO)
        job_parameters[1] = string(name: 'TORCH_COMMIT', value: baseline_TORCH_COMMIT)
        job_parameters[3] = string(name: 'backend', value: baseline_backend)
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

                echo "Job URL: ${BUILD_URL}/console .<br><br>" | tee ${WORKSPACE}/torch_clone.log
                cd ${WORKSPACE}
                git clone ${target_TORCH_REPO} target_pytorch
                cd ${WORKSPACE}/target_pytorch
                git checkout ${target_TORCH_COMMIT} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                result=$?
                
                if [ "${result}" -eq 0 ]; then
                    echo "<br><br>[INFO] Target torch repo and commit is correct.<br><br>" | tee -a ${WORKSPACE}/torch_clone.log
                else
                    echo "<br><br>[ERROR] Target torch repo and commit is wrong!<br><br>" | tee -a ${WORKSPACE}/torch_clone.log
                    exit 1
                fi

                cd ${WORKSPACE}
                git clone ${baseline_TORCH_REPO} baseline_pytorch
                cd ${WORKSPACE}/baseline_pytorch
                git checkout ${baseline_TORCH_COMMIT} 2>&1 | tee -a ${WORKSPACE}/torch_clone.log
                result=$?
                if [ "${result}" -eq 0 ]; then
                    echo "<br><br>[INFO] Baseline torch repo and commit is correct.<br><br>" | tee -a ${WORKSPACE}/torch_clone.log
                else
                    echo "<br><br>[ERROR] Baseline torch repo and commit is wrong!<br><br>" | tee -a ${WORKSPACE}/torch_clone.log
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

    stage("Cache torch commit") {
        sh'''
            #!/usr/bin/env bash
            set -ex

            cd ${WORKSPACE}/target_pytorch
            commit_date=`git log -n 1 --format="%cs"`
            bref_commit=`git rev-parse --short HEAD`
            DOCKER_TAG="${commit_date}_${bref_commit}"
            echo "${DOCKER_TAG}" > ${WORKSPACE}/docker_image_tag_target.log

            cd ${WORKSPACE}/baseline_pytorch
            commit_date=`git log -n 1 --format="%cs"`
            bref_commit=`git rev-parse --short HEAD`
            DOCKER_TAG="${commit_date}_${bref_commit}"
            echo "${DOCKER_TAG}" > ${WORKSPACE}/docker_image_tag_baseline.log
        '''
        stash includes: 'docker_image_tag_target.log', name: 'docker_image_tag_target'
        stash includes: 'docker_image_tag_baseline.log', name: 'docker_image_tag_baseline'
        archiveArtifacts  "docker_image_tag*.log"
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

    try{
        stage("Build target images") {
            retry(maxRetries){
                unstash 'docker_image_tag_target'
                sleep(60)
                def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/docker_image_tag_target.log''').toString().trim().replaceAll("\n","")
                def image_build_job = build job: 'inductor_images_local_py310', propagate: false, parameters: [             
                        [$class: 'StringParameterValue', name: 'PT_REPO', value: "${target_TORCH_REPO}"],
                        [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${target_TORCH_COMMIT}"],
                        [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
                ]

                def buildStatus = image_build_job.getResult()
                def cur_job_url = image_build_job.getAbsoluteUrl()
                if (buildStatus == hudson.model.Result.FAILURE) {
                    sh'''
                        echo "[FAILED] Docker image build Job URL: ${cur_job_url}/console .<br>" | tee ${WORKSPACE}/target_image_build.log
                    '''
                    throw new Exception("Target docker image build job failed")
                }
            }
        }
    } catch (Exception e) {
        sh'''
            #!/usr/bin/env bash
            set -ex
            echo "Job URL: ${BUILD_URL}/console .<br>" | tee -a ${WORKSPACE}/target_image_build.log
        '''
        emailext(
            subject: "Inductor TAS pipeline Pre-Check failed",
            mimeType: "text/html",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail,
            body: '${FILE, path="target_image_build.log"}'
        )
        archiveArtifacts "target_image_build.log"
        throw e
    }

    try{
        stage("Build baseline images") {
            retry(maxRetries){
                unstash 'docker_image_tag_baseline'
                sleep(60)
                def DOCKER_TAG = sh(returnStdout:true,script:'''cat ${WORKSPACE}/docker_image_tag_baseline.log''').toString().trim().replaceAll("\n","")
                def image_build_job = build job: 'inductor_images_local_py310', propagate: false, parameters: [             
                        [$class: 'StringParameterValue', name: 'PT_REPO', value: "${baseline_TORCH_REPO}"],
                        [$class: 'StringParameterValue', name: 'PT_COMMIT', value: "${baseline_TORCH_COMMIT}"],
                        [$class: 'StringParameterValue', name: 'tag', value: "${DOCKER_TAG}"],
                ]

                def buildStatus = image_build_job.getResult()
                def cur_job_url = image_build_job.getAbsoluteUrl()
                if (buildStatus == hudson.model.Result.FAILURE) {
                    sh'''
                        echo "[FAILED] Docker image build Job URL: ${cur_job_url}/console .<br>" | tee ${WORKSPACE}/baseline_image_build.log
                    '''
                    throw new Exception("Target docker image build job failed")
                }
            }
        }
    } catch (Exception e) {
        sh'''
            #!/usr/bin/env bash
            set -ex
            echo "Job URL: ${BUILD_URL}/console .<br>" | tee -a ${WORKSPACE}/baseline_image_build.log
        '''
        emailext(
            subject: "Inductor TAS pipeline Pre-Check failed",
            mimeType: "text/html",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail,
            body: '${FILE, path="baseline_image_build.log"}'
        )
        archiveArtifacts "baseline_image_build.log"
        throw e
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

        sh'''
            touch ${WORKSPACE}/inductor_pipeline_summary.csv
            echo "job_status,test_str,job_link" > ${WORKSPACE}/inductor_pipeline_summary.csv
        '''
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

                def cur_job_status = benchmark_job.getCurrentResult()
                def cur_job_url = benchmark_job.getAbsoluteUrl()
                withEnv(["cur_job_status=${cur_job_status}", "test_str=${test_str}", "cur_job_url=${cur_job_url}"]){
                sh'''
                    echo "${cur_job_status},${test_str},${cur_job_url}" >> ${WORKSPACE}/inductor_pipeline_summary.csv
                '''
                }
            } // job_list
        } // for
        parallel job_list
    } // stage
    
    stage("Archive artifacts") {
        archiveArtifacts artifacts: "inductor_pipeline_summary.csv", fingerprint: true
    }

    stage('Email') {
        def title_string = "TAS-Pipeline-${target_backend}-${precision}-${shape}-${wrapper}"
        withEnv(["title_string=${title_string}"]){
        sh'''
            python -c "import pandas as pd; pd.read_csv('inductor_pipeline_summary.csv').to_html('table.html', index=False, render_links=True)"
            cp html/0_css.html inductor_pipeline_summary.html
            echo "<h2><a href='${BUILD_URL}/console'>${title_string}</a></h2>" >> inductor_pipeline_summary.html
            cat table.html >> inductor_pipeline_summary.html
        '''
        }
        archiveArtifacts artifacts: "inductor_pipeline_summary.html", fingerprint: true
        emailext(
            mimeType: "text/html",
            subject: title_string + "-summary_report",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail,
            body: '${FILE, path="inductor_pipeline_summary.html"}'
        )
    } 
   
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