env.NODE_LABEL = 'OOB-MCR'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

env.NODE_PROXY = 'http://proxy.ims.intel.com:911'
if ('NODE_PROXY' in params) {
    echo "NODE_PROXY in params"
    if (params.NODE_PROXY != '') {
        env.NODE_PROXY = params.NODE_PROXY
    }
}
echo "NODE_PROXY: $NODE_PROXY"

env.modelids = 'meta-llama/Llama-2-7b-hf'
if ('modelids' in params) {
    echo "modelids in params"
    if (params.modelids != '') {
        env.modelids = params.modelids
        modelids = modelids.split(",")
    }
}
echo "modelids: $modelids"

env.dtypes = 'bfloat16'
if ('dtypes' in params) {
    echo "dtypes in params"
    if (params.dtypes != '') {
        env.dtypes = params.dtypes
        dtypes = dtypes.split(",")
    }
}
echo "dtypes: $dtypes"

env.pt_repo = 'https://github.com/pytorch/pytorch.git'
if ('pt_repo' in params) {
    echo "pt_repo in params"
    if (params.pt_repo != '') {
        env.pt_repo = params.pt_repo
    }
}
echo "pt_repo: $pt_repo"

env.pt_branch= 'main'
if ('pt_branch' in params) {
    echo "pt_branch in params"
    if (params.pt_branch != '') {
        env.pt_branch = params.pt_branch
    }
}
echo "pt_branch: $pt_branch"

env.xpu_ops_repo = 'https://github.com/intel/torch-xpu-ops.git'
if ('xpu_ops_repo' in params) {
    echo "xpu_ops_repo in params"
    if (params.xpu_ops_repo != '') {
        env.xpu_ops_repo = params.xpu_ops_repo
    }
}
echo "xpu_ops_repo: $xpu_ops_repo"

env.xpu_ops_branch= 'main'
if ('xpu_ops_branch' in params) {
    echo "xpu_ops_branch in params"
    if (params.xpu_ops_branch != '') {
        env.xpu_ops_branch = params.xpu_ops_branch
    }
}
echo "xpu_ops_branch: $xpu_ops_branch"

env.backend= 'compile'
if ('backend' in params) {
    echo "backend in params"
    if (params.backend != '') {
        env.backend = params.backend
        backend = backend.split(",")
    }
}
echo "backend: $backend"

env.profile= 'false'
if ('profile' in params) {
    echo "profile in params"
    if (params.profile != '') {
        env.profile = params.profile
    }
}
echo "profile: $profile"

env.device= 'cpu'
if ('device' in params) {
    echo "device in params"
    if (params.device != '') {
        env.device = params.device
    }
}
echo "device: $device"

env.extra_args= ''
if ('extra_args' in params) {
    echo "extra_args in params"
    if (params.extra_args != '') {
        env.extra_args = params.extra_args
    }
}
echo "extra_args: $extra_args"

env.input_length = ''
if ('input_length' in params) {
    echo "input_length"
    if (params.input_length != '') {
        input_length = params.input_length
        input_length = input_length.split(",")
    }
}
echo "input_length: $input_length"

env.iter = '5'
if ('iter' in params) {
    echo "iter"
    if (params.iter != '') {
        iter = params.iter
    }
}
echo "iter: $iter"

env.groupsize = ''
if ('groupsize' in params) {
    echo "groupsize"
    if (params.groupsize != '') {
        groupsize = params.groupsize
        groupsize = groupsize.split(",")
    }
}
echo "groupsize: $groupsize"

env.output_length = ''
if ('output_length' in params) {
    echo "output_length"
    if (params.output_length != '') {
        output_length = params.output_length
        output_length = output_length.split(",")
    }
}
echo "output_length: $output_length"

env.build_image= 'False'
if ('build_image' in params) {
    echo "build_image in params"
    if (params.build_image != '') {
        env.build_image = params.build_image
    }
}
echo "build_image: $build_image"

env.docker_name= 'test'
if ('docker_name' in params) {
    echo "docker_name in params"
    if (params.docker_name != '') {
        env.docker_name = params.docker_name
    }
}
echo "docker_name: $docker_name"

env.upload_log= 'False'
if ('upload_log' in params) {
    echo "upload_log in params"
    if (params.upload_log != '') {
        env.upload_log = params.upload_log
    }
}
echo "upload_log: $upload_log"

env.tensor_parallel= 'False'
if ('tensor_parallel' in params) {
    echo "tensor_parallel in params"
    if (params.tensor_parallel != '') {
        env.tensor_parallel = params.tensor_parallel
    }
}
echo "tensor_parallel: $tensor_parallel"

env.tp_sockets= '1'
if ('tp_sockets' in params) {
    echo "tp_sockets in params"
    if (params.tp_sockets != '') {
        env.tp_sockets = params.tp_sockets
        tp_sockets = tp_sockets.split(",")
    }
}
echo "tp_sockets: $tp_sockets"

env.hardware= 'emr'
if ('hardware' in params) {
    echo "hardware in params"
    if (params.hardware != '') {
        env.hardware = params.hardware
    }
}
echo "hardware: $hardware"

env.test_mode= 'performance'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        env.test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"

env.docker_pull= 'True'
if ('docker_pull' in params) {
    echo "docker_pull in params"
    if (params.docker_pull != '') {
        env.docker_pull = params.docker_pull
    }
}
echo "docker_pull: $docker_pull"

env.torchtune_modeldir= '/localdisk/datasets/huggingface/'
if ('torchtune_modeldir' in params) {
    echo "torchtune_modeldir in params"
    if (params.torchtune_modeldir != '') {
        env.torchtune_modeldir = params.torchtune_modeldir
    }
}
echo "torchtune_modeldir: $torchtune_modeldir"

env.http_proxy=env.NODE_PROXY
env.https_proxy=env.NODE_PROXY
env.LOG_DIR = 'distributed_log'

node(NODE_LABEL){
    stage("Prepare Stock Pytorch"){
        println('prepare......')
        // TODO: implement report_only logic
        deleteDir()
        retry(3){
            checkout([
                $class: 'GitSCM',
                branches: scm.branches,
                doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                extensions: scm.extensions + [cloneOption(depth: 1, honorRefspec: true, noTags: true, reference: '', shallow: true, timeout: 10)],
                userRemoteConfigs: scm.userRemoteConfigs
            ])
        }
        sh'''
            #!/usr/bin/env bash
            mkdir -p ${WORKSPACE}/${LOG_DIR}
            conda create -n ${conda_name} python=${python_version} cmake=3.28 ninja -y
            conda activate ${conda_name}
            git clone ${pt_repo} pytorch
            cd pytorch && git checkout ${pt_branch}
            git submodule sync && git submodule update --init --recursive
            sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt
            rm -rf third_party/torch-xpu-ops
            cd third_party
            git clone ${xpu_ops_repo} torch-xpu-ops
            cd torch-xpu-ops
            git checkout ${xpu_ops_branch}
        '''
    }
   
    stage("Build Pytorch XPU"){
        echo 'Building PyTorch......'
        sh '''
        set -xe
        conda activate ${conda_name}
        source scripts/modelbench/distributed/env.sh
        export USE_XCCL=1
        cd pytorch
        pip install -r requirements.txt
        current_commit=$(git rev-parse HEAD)
        WERROR=1 python setup.py bdist_wheel 2>&1 | tee ${WORKSPACE}/${LOG_DIR}/pytorch_${current_commit}_build.log
        pip install --force-reinstall dist/*.whl
        git clone https://github.com/pytorch/vision && cd vision && python setup.py install && cd ..
        '''
    }

    stage("Run Torch XPU Distributed UT"){
        echo "Running distributed UT"
        sh '''
        set -xe
        conda activate ${conda_name}
        source scripts/modelbench/distributed/env.sh
        pip install pytest pytest-timeout
        sudo cp /proc/sys/kernel/yama/ptrace_scope ptrace_scope.bk
        sudo echo "0"|sudo tee /proc/sys/kernel/yama/ptrace_scope
        cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu
        XCCL_EANBLE=$(python -c "import torch;print(torch.distributed.is_xccl_available())")
        if [[ "${XCCL_ENABLE}}" == 'False' ]]; then
            echo -e "[ERROR] XCCL is not enabled"
            exit 1
        fi
        python run_distributed_local.py 2>&1 | tee ${WORKSPACE}/${LOG_DIR}/pytorch_distributed_test.log
        cd ${WORKSPACE}
        sudo cp ptrace_scope.bk /proc/sys/kernel/yama/ptrace_scope
        '''
    }
    
    stage("Archive logs"){
    if (fileExists("${WORKSPACE}/logs/summary.log") == true){
        archiveArtifacts "logs/summary.log"
    }   
    archiveArtifacts artifacts: "**/distributed_log/**", excludes: null
    fingerprint: true 
    }  
}

