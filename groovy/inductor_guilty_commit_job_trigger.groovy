node("weizhuoz-mlt-ace"){
    checkout scm
    stage('Trigger Guilty Commit Job'){
        def lines = readJSON file: 'test.json'
        int size = lines.size();
        println(size)
        def job_list = [:]
        for (i = 0; i < size; i += 1) {
            def elem = lines[i]
            if (test_kinds.contains(elem['kind'])) {
                job_list["Job_${i}"] = {
                    guilty_commit_search_job = build job: 'inductor_aws_guilty_commit_search_zwz', propagate: false, parameters: [
                        [$class: 'StringParameterValue', name: 'default_mail', value: default_mail],
                        [$class: 'StringParameterValue', name: 'precision', value: elem['precision']],                
                        [$class: 'StringParameterValue', name: 'shape', value: shape],
                        [$class: 'StringParameterValue', name: 'WRAPPER', value: "default"],
                        [$class: 'StringParameterValue', name: 'TORCH_BRANCH', value: TORCH_BRANCH],
                        [$class: 'StringParameterValue', name: 'THREADS', value: elem['thread']],
                        [$class: 'StringParameterValue', name: 'instance_name', value: elem['instance_name']],
                        [$class: 'StringParameterValue', name: 'suite', value: elem['suite']],
                        [$class: 'StringParameterValue', name: 'model', value: elem['name']],
                        [$class: 'StringParameterValue', name: 'scenario', value: elem['scenario']],
                        [$class: 'StringParameterValue', name: 'kind', value: elem['kind']],
                        [$class: 'StringParameterValue', name: 'TORCH_START_COMMIT', value: TORCH_START_COMMIT],
                        [$class: 'StringParameterValue', name: 'TORCH_END_COMMIT', value: TORCH_END_COMMIT],
                    ]
                }
            }
        }
        parallel job_list
    }
}