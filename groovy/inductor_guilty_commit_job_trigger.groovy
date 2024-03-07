env.devloper_email = "guobing.chen@intel.com;beilei.zheng@intel.com;xuan.liao@intel.com;chunyuan.wu@intel.com;haozhe.zhu@intel.com;weiwen.xia@intel.com;jiong.gong@intel.com;shufan.wu@intel.com;diwei.sun@intel.com;leslie.fang@intel.com;mengfei.li@intel.com"

node(NODE_LABEL){
    deleteDir()
    checkout scm
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
            echo "suite,model,scenario,thread,kind,precision,shape,wrapper,guilty_commit,job_link" > ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
        '''
        def common_info_dict = readJSON file: 'guilty_commit_search_common_info.json'
        def model_list_lines = readJSON file: 'guilty_commit_search_model_list.json'

        int size = model_list_lines.size();
        def job_list = [:]
        Set model_set = []
        for (i = 0; i < size; i += 1) {
            def elem = model_list_lines[i]
            if (test_kinds.contains(elem['kind']) && (test_thread.contains(elem['thread']))) {
                // For fixed model, only test one thread_mode
                def filter_kind_list = ["fixed"]
                if (filter_kind_list.contains(elem['kind'])) {
                    if (model_set.contains(elem['name'])) {
                        continue
                    } else {
                        model_set.add(elem['name'])
                    }
                }
                
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
                    def path = 'inductor_guilty_commit_search/'+ cur_job_number + '/**/inductor_log/guilty_commit.log'
                    def log_dir = 'inductor_guilty_commit_search/'+ cur_job_number + '/**/inductor_log'
                    def temp_files = findFiles glob: path
                    println(path)
                    println(temp_files.length)

                    copyArtifacts(
                        projectName: guilty_commit_search_job_name,
                        selector: specific("${cur_job_number}"),
                        target: "inductor_guilty_commit_search/${cur_job_number}"
                    )

                    withEnv([
                        "cur_job_number=${cur_job_number}",
                        "suite=${suite}",
                        "model=${model}",
                        "scenario=${scenario}",
                        "thread=${thread}",
                        "kind=${kind}",
                        "precision=${precision}",
                        "shape=${shape}",
                        "wrapper=${wrapper}",
                        "cur_job_url=${cur_job_url}",
                        "file_path=${path}",
                        "log_dir=${log_dir}"
                    ]) {
                        def files = findFiles glob: path
                        println(files.length)
                        sh'''
                            ls -l ${log_dir}
                        '''
                        if (files.length > 0){
                            sh'''
                                guilty_commit=`cat ${file_path} | head -1`
                                echo "${suite},${model},${scenario},${thread},${kind},${precision},${shape},${wrapper},${guilty_commit},${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                            '''
                        } else {
                            sh'''
                                echo "${suite},${model},${scenario},${thread},${kind},${precision},${shape},${wrapper},fake,${cur_job_url}" >> ${WORKSPACE}/inductor_guilty_commit_search_summary.csv
                            '''
                        }
                    }
                }
            }
        } // for
        parallel job_list
        
    } // stage trigger

    stage("archiveArtifacts") {
        archiveArtifacts artifacts: "**/inductor_guilty_commit_search/**", fingerprint: true
        archiveArtifacts artifacts: "inductor_guilty_commit_search_summary.csv", fingerprint: true
    }

    stage('Email') {
        sh'''
            python -c "import pandas as pd; pd.read_csv('inductor_guilty_commit_search_summary.csv').to_html('table.html', index=False, render_links=True)"
            cp html/0_css.html inductor_guilty_commit_search_summary.html
            echo "<h1><a href='${BUILD_URL}'>Job Link</a></h1>" >> inductor_guilty_commit_search_summary.html
            cat table.html >> inductor_guilty_commit_search_summary.html
        '''
        archiveArtifacts artifacts: "inductor_guilty_commit_search_summary.html", fingerprint: true
        emailext(
            mimeType: "text/html",
            subject: "Torchinductor-Auto_guilty_commit_search_summary_report",
            from: "pytorch_inductor_val@intel.com",
            to: default_mail + ";" + devloper_email,
            body: '${FILE, path="inductor_guilty_commit_search_summary.html"}'
        )
    }
}