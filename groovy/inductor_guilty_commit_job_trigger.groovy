def job_list = [:]

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
        def common_info_dict = readJSON file: 'guilty_commit_search_common_info.json'
        def model_list_lines = readJSON file: 'guilty_commit_search_model_list.json'

        int size = model_list_lines.size();
        for (i = 0; i < size; i += 1) {
            def elem = model_list_lines[i]
            if (test_kinds.contains(elem['kind'])) {
                def instance_name = 'icx-guilty-search'
                if (elem['precision'] == "amp") {
                    instance_name = 'spr-guilty-search'
                }

                job_parameters = [
                    string(name: 'default_mail', value: default_mail),
                    string(name: 'precision', value: elem['precision']),
                    string(name: 'shape', value: common_info_dict['shape']),
                    string(name: 'WRAPPER', value: common_info_dict['wrapper']),
                    string(name: 'TORCH_REPO', value: common_info_dict['torch_repo']),
                    string(name: 'TORCH_BRANCH', value: common_info_dict['torch_branch']),
                    string(name: 'THREADS', value: elem['thread']),
                    string(name: 'instance_name', value: instance_name),
                    string(name: 'suite', value: elem['suite']),
                    string(name: 'model', value: elem['name']),
                    string(name: 'scenario', value: elem['scenario']),
                    string(name: 'kind', value: elem['kind']),
                    string(name: 'TORCH_START_COMMIT', value: common_info_dict['start_commit']),
                    string(name: 'TORCH_END_COMMIT', value: common_info_dict['end_commit']),
                ]

                job_list["job_${i}"] = {
                    build job: guilty_commit_search_job_name,
                    propagate: false,
                    parameters: job_parameters,
                }
            }
        }
        parallel job_list
    }

    stage('Collecting and Analyzing report'){
        sh'''
            touch ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
            echo "suite,name,scenario,thread,kind,precision,guilty_commit,job_link" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
        '''
        for (key in job_list) {
            job_variables = job_list[key].getBuildVariables()
            suite = job_variables['suite']
            model = job_variables['model']
            scenario = job_variables['scenario']
            thread = job_variables['THREADS']
            kind = job_variables['kind']
            precision = job_variables['precision']

            cur_job_number = job_list[key].getNumber()
            cur_job_url = job_list[key].getAbsoluteUrl()
            cur_job_duration = job_list[key].getDurationString()

            copyArtifacts(
                projectName: guilty_commit_search_job_name,
                selector: specific("${cur_job_number}"),
                target: "inductor_guilty_commit_search/${cur_job_number}"
            )
            if (fileExists("${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/guilty_commit.log") == true){
                sh'''
                    guilty_commit=`cat ${WORKSPACE}/inductor_guilty_commit_search/${cur_job_number}/*/inductor_log/guilty_commit.log | head -1`
                    echo "${suite},${name},${scenario},${thread},${kind},${precision},${guilty_commit},${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                '''
            } else {
                sh'''
                    echo "${suite},${name},${scenario},${thread},${kind},${precision},fake,${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                '''
            }
        }
    }

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