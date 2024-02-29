def job_list = [:]
def job_params_list = [:]

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
                job_params_list["job_${i}"] = [
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

                job_list["job_${i}"] = {guilty_commit_search_job = build propagate: false, job: guilty_commit_search_job_name, parameters: job_params_list["job_${i}"]}
            }
        }
        parallel job_list
    }

    stage('Collecting and Analyzing report'){
        sh'''
            touch ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
            echo "suite,name,scenario,thread,kind,precision,guilty_commit,job_link" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
        '''
        int size = job_list.size()
        for (i = 0; i < size; i += 1) {
            suite = job_params_list["job_${i}"]['suite']
            name = job_params_list["job_${i}"]['name']
            scenario = job_params_list["job_${i}"]['scenario']
            thread = job_params_list["job_${i}"]['THREADS']
            kind = job_params_list["job_${i}"]['kind']
            precision = job_params_list["job_${i}"]['precision']

            cur_job_number = job_list["job_${i}"].getNumber()
            cur_job_url = job_list["job_${i}"].getAbsoluteUrl()
            cur_job_duration = job_list["job_${i}"].getDurationString()

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