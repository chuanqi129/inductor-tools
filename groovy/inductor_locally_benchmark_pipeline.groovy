def str_list = ['target', 'baseline']
env.benchmark_job = 'inductor_locally_benchmark'
env.result_compare_job = 'inductor_job_result_compare'
env.target_job_selector = 'None'
env.baseline_job_selector = 'None'

def getJobParameters(String test_str) {
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
    ]
    if (test_str == "baseline") {
        println("[INFO]: baseline pytorch repo and commit: ")
        job_parameters[0] = string(name: 'TORCH_REPO', value: baseline_TORCH_REPO)
        job_parameters[1] = string(name: 'TORCH_COMMIT', value: baseline_TORCH_COMMIT)
    }
    return job_parameters
}

node(report_node){
    deleteDir()
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

    stage("Pre-check repo and commit") {
        if ((target_TORCH_REPO == baseline_TORCH_REPO) && 
            (target_TORCH_COMMIT == baseline_TORCH_COMMIT)) {
            println("same repo and commit")
        }
    }

    stage('Trigger aws benchmark Job'){
        def job_list = [:]
        for (sub_str in str_list) {
            def test_str = sub_str
            job_list[test_str] = {
                def job_parameters = getJobParameters(test_str)
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