// set parameters
properties([
    parameters([
        string(name: 'NODE_LABEL', defaultValue: 'clx137', description: '', trim: true),
        string(name: 'create_conda_env', defaultValue: "true", description: '', trim: true),
        string(name: 'conda_env_name', defaultValue: 'pt_win', description: '', trim: true),
        string(name: 'recipients', defaultValue: 'lifeng.a.wang@intel.com', description: '', trim: true),
    ])
])

node(NODE_LABEL) {
    stage("prepare the environment") {
        deleteDir()
        checkout scm
        if(params.create_conda_env)
        {
            pwsh '''
            cmd.exe "/K" (
            '"C:/Program Files (x86)/Intel/oneAPI/setvars.bat" ' +
            '&& pwsh -File inductor-tools/scripts/windows_inductor/prepare_env_nightly.ps1 ' +
            '-envName ${conda_env_name}'
            )
            '''
        }

        pwsh '''
        Set-Location "$env:WORKSPACE"
        git clone --depth=1 https://github.com/pytorch/pytorch.git
        '''
    }
}
