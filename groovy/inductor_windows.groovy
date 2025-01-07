// set parameters
properties([
    parameters([
        string(name: 'NODE_LABEL', defaultValue: 'clx137', description: '', trim: true),
        bool(name: 'create_conda_env', defaultValue: true, description: '', trim: true),
        string(name: 'conda_env_name', defaultValue: 'pt_win', description: '', trim: true),
        string(name: 'recipients', defaultValue: 'lifeng.a.wang@intel.com', description: '', trim: true),
    ])
])

node(NODE_LABEL) {
    stage("prepare the environment") {
        deleteDir()
        checkout scm
        pwsh '''
        Set-Location "$env:WORKSPACE"
        cmd.exe "/K" (
            '"C:/Program Files (x86)/Intel/oneAPI/setvars.bat" ' +
            '&& pwsh -File inductor-tools/scripts/windows_inductor/prepare_env_nightly.ps1 ' +
            '-create_conda_env ${create_conda_env} ' +
            '-envName ${conda_env_name}'
        )
        '''
    }
}
