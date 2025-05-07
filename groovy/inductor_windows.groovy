// set parameters
properties([
    parameters([
        string(name: 'NODE_LABEL', defaultValue: 'clx137', description: '', trim: true),
        booleanParam(name: 'create_conda_env', defaultValue: true, description: ''),
        string(name: 'conda_env_name', defaultValue: 'pt_win', description: '', trim: true),
        choice(name: 'compiler', choices: ['msvc', 'icc'], description: ''),
        choice(name: 'wrapper', choices: ['cpp', 'default'], description: ''),
        choice(name: 'suite', choices: ['all', 'torchbench', 'huggingface', 'timm_models'], description: ''),
        choice(name: 'precision', choices: ['float32'], description: ''),
        string(name: 'recipients', defaultValue: 'chuanqi.wang@intel.com;yucheng.liu@intel.com;leslie.fang@intel.com;xu.han@intel.com;lifeng.a.wang@intel.com', description: '', trim: true),
        string(name: 'http_proxy', defaultValue: 'http://proxy.ims.intel.com:911', description: '', trim: true),
        choice(name: 'refer_build', choices: ['lastSuccessfulBuild', '0'], description: 'choose 0 when you need only this round report'),
    ])
])

node(NODE_LABEL) {
    stage("download the repositories") {
        deleteDir()
        checkout scm

        retry(3) {
        // Clones the PyTorch/Benchmark repository from GitHub with a depth of 1, which means only the latest commit will be cloned.
        pwsh """
        \$env:HTTP_PROXY = "${http_proxy}"
        \$env:HTTPS_PROXY = "${http_proxy}"
        git clone --depth=1 https://github.com/pytorch/pytorch.git
        """
        }
    }

    stage("prepare the conda environment") {
        if(params.create_conda_env)
            /**
             * Executes a PowerShell script to set up the environment for the inductor tools on Windows.
             *
             * The script performs the following steps:
             * 1. Runs the Intel oneAPI setvars.bat script to set up the Intel environment variables.
             * 2. Executes a PowerShell script to prepare the environment with nightly builds.
             *
             * @param conda_env_name The name of the conda environment to be used.
             */
            {
                pwsh """
                \$env:HTTP_PROXY = "${http_proxy}"
                \$env:HTTPS_PROXY = "${http_proxy}"
                cmd.exe "/K" (
                '"C:/Program Files (x86)/Intel/oneAPI/setvars.bat" ' +
                '&& pwsh -File scripts/windows_inductor/prepare_env_nightly.ps1 ' +
                '-envName ${conda_env_name}'
                )
                """
            }
    }

    stage('conduct the benchmarks'){
        def workspaceDir = env.WORKSPACE
        def logsDir = "${workspaceDir}/inductor_log"
        pwsh """
        \$env:HTTP_PROXY = "${http_proxy}"
        \$env:HTTPS_PROXY = "${http_proxy}"
        New-Item -ItemType SymbolicLink -Path "${workspaceDir}\\benchmark" -Target "C:\\benchmark"
        Set-Location pytorch
        cmd.exe "/K" (
            '"C:/Program Files (x86)/Intel/oneAPI/setvars.bat" ' +
            '&& pwsh -File ../scripts/windows_inductor/dynamo_test.ps1 -dir ${logsDir} ' +
            '-envName ${conda_env_name} -suite ${suite} -precision ${precision} -compiler ${compiler}'
        )
        """
        archiveArtifacts artifacts: 'inductor_log/**', fingerprint: true
    }

    stage('generate the report'){
        if(refer_build != '0') {
            copyArtifacts(
                projectName: currentBuild.projectName,
                selector: specific("${refer_build}"),
                fingerprintArtifacts: true,
                target: "refer",
            )
            pwsh """
            \$env:HTTP_PROXY = "${http_proxy}"
            \$env:HTTPS_PROXY = "${http_proxy}"
            conda run -n $conda_env_name pip install styleframe
            conda run -n $conda_env_name python.exe scripts/windows_inductor/report_win.py -p $precision -m inference -sc accuracy performance -s $suite -r refer
            """
        }else{     
            pwsh """
            \$env:HTTP_PROXY = "${http_proxy}"
            \$env:HTTPS_PROXY = "${http_proxy}"
            conda run -n $conda_env_name pip install styleframe
            conda run -n $conda_env_name python.exe scripts/windows_inductor/report_win.py -p $precision -m inference -sc accuracy performance -s $suite
            """
        }
        archiveArtifacts artifacts: 'inductor_log/Inductor_E2E_Test_Report.xlsx', fingerprint: true
    }

    stage('send email'){
        emailext body: 'Please check the attachment for the inductor report.',
            subject: "[Regular Weekly]-Windows-Inductor-Test-${env.compiler}-${env.wrapper}-Report",
            to: params.recipients,
            attachmentsPattern: 'inductor_log/Inductor_E2E_Test_Report.xlsx'
    }
}
