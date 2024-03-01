node(NODE_LABEL){
    checkout scm
    deleteDir()
    stage("Copy Artifacts"){
        copyArtifacts(
            projectName: "${target_job}",
            selector: specific("${target_job_selector}"),
            fingerprintArtifacts: true
        ) 
    }

    stage('Trigger Guilty Commit Job'){
        sh'''
            touch ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
            echo "suite,name,scenario,thread,kind,precision,shape,wrapper,guilty_commit,job_link" > ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
        '''
        def common_info_dict = readJSON file: 'guilty_commit_search_common_info.json'
        def model_list_lines = readJSON file: 'guilty_commit_search_model_list.json'

        int size = model_list_lines.size();
        def job_list = [:]
        for (i = 0; i < size; i += 1) {
            def elem = model_list_lines[i]
            if (test_kinds.contains(elem['kind'])) {
                def instance_name = 'icx-guilty-search'
                if (elem['precision'] == "amp") {
                    instance_name = 'spr-guilty-search'
                }

                def suite = elem['suite']
                def model = elem['name']
                def scenario = elem['scenario']
                def thread = elem['thread']
                def kind = elem['kind']
                def precision = elem['precision']
                def shape = common_info_dict['shape']
                def wrapper = common_info_dict['wrapper']

                job_list["${suite}_${model}_${precision}_${kind}_${thread}"] = {
                    def job_parameters = [
                        string(name: 'default_mail', value: default_mail),
                        string(name: 'suite', value: suite),
                        string(name: 'model', value: model),
                        string(name: 'scenario', value: scenario),
                        string(name: 'THREADS', value: thread),
                        string(name: 'kind', value: kind),
                        string(name: 'precision', value: precision),
                        string(name: 'shape', value: shape),
                        string(name: 'WRAPPER', value: wrapper),
                        string(name: 'instance_name', value: instance_name),
                        string(name: 'TORCH_REPO', value: common_info_dict['torch_repo']),
                        string(name: 'TORCH_BRANCH', value: common_info_dict['torch_branch']),
                        string(name: 'TORCH_START_COMMIT', value: common_info_dict['start_commit']),
                        string(name: 'TORCH_END_COMMIT', value: common_info_dict['end_commit']),
                    ]

                    def guilty_commit_job = build propagate: false,
                        job: guilty_commit_search_job_name, parameters: job_parameters
                    
                    def cur_job_number = guilty_commit_job.getNumber()
                    def cur_job_url = guilty_commit_job.getAbsoluteUrl()
                    def cur_job_duration = guilty_commit_job.getDurationString()

                    copyArtifacts(
                        projectName: guilty_commit_search_job_name,
                        selector: specific("${cur_job_number}"),
                        target: "inductor_guilty_commit_search/${cur_job_number}"
                    )

                    withEnv([
                        "cur_job_number=${cur_job_number}",
                        "suite=${suite}",
                        "name=${name}",
                        "scenario=${scenario}",
                        "thread=${thread}",
                        "kind=${kind}",
                        "precision=${precision}",
                        "shape=${shape}",
                        "wrapper=${wrapper}",
                        "cur_job_url=${cur_job_url}",
                    ]) {
                        if (fileExists("${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/perf_drop.log") == true){
                            sh'''
                                cat ${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/perf_drop.log
                            '''
                        }
                        if (fileExists("${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/guilty_commit.log") == true){
                            sh'''
                                guilty_commit=`cat ${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/guilty_commit.log | head -1`
                                echo "${suite},${name},${scenario},${thread},${kind},${precision},${shape},${wrapper},${guilty_commit},${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                            '''
                        } else {
                            sh'''
                                echo "${suite},${name},${scenario},${thread},${kind},${precision},${shape},${wrapper},fake,${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                            '''
                        }
                    }
                }
            }
        } // for
        parallel job_list
    } // stage trigger

    // stage('Email') {
    //     emailext(
    //         subject: "Torchinductor-"
    //         mimeType: "text/html",
    //         from: "pytorch_inductor_val@intel.com",
    //         to: default_mail,
    //         body: '${FILE, path="html/guilty_commit_search_summary.html"}'
    //     )
    // }
}