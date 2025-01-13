// set parameters
properties([
    parameters([
        string(name: 'NODE_LABEL', defaultValue: 'clx137', description: '', trim: true),
        booleanParam(name: 'create_conda_env', defaultValue: true, description: ''),
        string(name: 'conda_env_name', defaultValue: 'pt_win', description: '', trim: true),
        choice(name: 'compiler', choices: ['msvc', 'icc'], description: ''),
        choice(name: 'wrapper', choices: ['default', 'cpp'], description: ''),
        choice(name: 'suite', choices: ['all', 'torchbench', 'huggingface', 'timm_models'], description: ''),
        choice(name: 'precision', choices: ['float32'], description: ''),
        string(name: 'recipients', defaultValue: 'lifeng.a.wang@intel.com', description: '', trim: true),
    ])
])

node(NODE_LABEL) {
    stage("download the repositories") {
        deleteDir()
        checkout scm

        retry(3) {
        // Clones the PyTorch/Benchmark repository from GitHub with a depth of 1, which means only the latest commit will be cloned.
        pwsh '''
        $env:HTTP_PROXY = "http://proxy.ims.intel.com:911"
        $env:HTTPS_PROXY = "http://proxy.ims.intel.com:911"  # add it to the jenkins node
        git clone --depth=1 https://github.com/pytorch/pytorch.git
        git clone --depth=1 https://github.com/pytorch/benchmark.git
        '''
        }
    }

    stage("prepare the conda environment") {
        if(params.create_conda_env)
            /**
             * Executes a PowerShell script to set up the environment for the inductor tools on Windows.
             *
             * The script performs the following steps:
             * 1. Runs the Intel oneAPI setvars.bat script to set up the Intel environment variables.
             * 2. Executes a PowerShell script to prepare the environment for nightly builds.
             *
             * @param conda_env_name The name of the conda environment to be used.
             */
            {
                pwsh """
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
        stage('generate the report'){
        pwsh """
        conda run -n $conda_env_name pip install styleframe
        conda run -n $conda_env_name python.exe scripts/windows_inductor/report_win.py -p $precision -m inference -sc accuracy performance -s $suite
        """
    }
        archiveArtifacts artifacts: 'inductor_log/Inductor_E2E_Test_Report.xlsx', fingerprint: true
    }

    stage('send email'){
        emailext body: 'Please check the attachment for the inductor report.',
            subject: 'Windows Inductor Report',
            to: params.recipients,
            attachmentsPattern: 'inductor_log/Inductor_E2E_Test_Report.xlsx'
    }
}
