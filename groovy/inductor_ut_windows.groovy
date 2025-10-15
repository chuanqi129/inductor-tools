node(NODE_LABEL) {
    stage("download the repositories") {
        deleteDir()
        checkout scm
    }

    stage("prepare the conda environment") {
        if(params.create_conda_env)
            /**
             * Prepare the conda environment and install the nightly PyTorch build
             */
            {
                pwsh """
                \$env:HTTP_PROXY = "${http_proxy}"
                \$env:HTTPS_PROXY = "${http_proxy}"
                pwsh -File scripts/windows_inductor/install_nightly_pytorch.ps1 -envName ${conda_env_name}
                """
            }
    }

    stage('conduct the benchmarks'){
        def workspaceDir = env.WORKSPACE
        def logsDir = "${workspaceDir}/inductor_log"
        pwsh """
        \$env:HTTP_PROXY = "${http_proxy}"
        \$env:HTTPS_PROXY = "${http_proxy}"
        Set-Location C:/pytorch
        git checkout nightly
        git pull
        Copy-Item -Path "${workspaceDir}/scripts/windows_inductor/cpu_ut.ps1" -Destination "."

        cmd.exe "/K" (
            '"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"' +
            '&& pwsh -File cpu_ut.ps1 -log_dir ${logsDir} -envName ${conda_env_name} '
        )
        """
        archiveArtifacts artifacts: 'inductor_log/**', fingerprint: true
    }

}
