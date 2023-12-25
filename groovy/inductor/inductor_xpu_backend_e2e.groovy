node(env.nodes_label){
    cleanWs()
    stage('setup env') {
        println('================================================================')
        println('setup nodes env')
        println('================================================================')
        sh '''
        set -e
        set +x
        if [ -e ${HOME}/miniconda3/etc/profile.d/conda.sh ];then
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            if [ -d ${HOME}/miniconda3/envs/${conda_env} ];then
                conda activate ${conda_env}
            else
                conda create -n ${conda_env} cmake ninja pillow python=${python_version}
            fi
        else
            wget -q -e https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            bash Miniconda3-latest-Linux-x86_64.sh -b
            source ${HOME}/miniconda3/bin/activate
            conda create -n ${conda_env} cmake ninja pillow python=${python_version}
        fi
        '''
    }//stage
    stage('Install Dependency') {
        println('================================================================')
        println('Install Dependency')
        println('================================================================')
            checkout scm
            sh '''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh ${oneapi_ver}
            python -m pip install numpy --upgrade --no-cache --force-reinstall

            if [[ -n ${torch_whl} ]] && [[ -n ${ipex_whl} ]];then
                python -m pip uninstall -y torch || true
                wget -q -e use_proxy=no ${torch_whl}
                python -m pip install --force-reinstall $(basename ${torch_whl})
                python -m pip uninstall -y intel_extension_for_pytorch || true
                wget -q -e use_proxy=no ${ipex_whl}
                python -m pip install --force-reinstall $(basename ${ipex_whl})
            else
                bash ${WORKSPACE}/scripts/inductor/env_prepare.sh
                source ${HOME}/env.sh ${oneapi_ver}
                python -c "import torch;import intel_extension_for_pytorch"
                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    echo -e "[ERROR] Public-torch or IPEX BUILD FAIL"
                    exit 1
                fi
            fi
            '''
    }//stage
    stage('Build Triton') {
        println('================================================================')
        println('Build Triton')
        println('================================================================')
        retry(1){
            sh'''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh ${oneapi_ver}
            source ${HOME}/set_proxy.sh
            pip uninstall -y triton
            sudo update-ca-certificates --fresh
            export SSL_CERT_DIR=/etc/ssl/certs
            pip install pybind11
            if [[ -n ${triton_whl} ]] ;then
                pip uninstall -y triton || true
                wget -q -e use_proxy=no ${triton_whl}
                python -m pip install --force-reinstall $(basename ${triton_whl})
            else
                git clone https://github.com/openai/triton triton
                cd triton
                git submodule sync
                git submodule update --init --recursive --jobs 0
                cd third_party/intel_xpu_backend
                git checkout main && git pull
                cd ../../python
                python setup.py clean
                TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py bdist_wheel
                pip install dist/*.whl
                cd ${WORKSPACE}/triton
                python -c "import triton"
            fi
            '''
        }//retry
    }//stage
    stage('One-by-One-Test'){
        println('================================================================')
        println('One-by-One-Test')
        println('================================================================')
        if(env.skip_OBO != 'True'){
            try {
                sh'''
                set -e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh ${oneapi_ver}
                source ${HOME}/set_proxy.sh
                mkdir -p ${WORKSPACE}/logs
                if [ ! -d ${WORKSPACE}/frameworks.ai.pytorch.private-gpu ];then
                    git clone -b ${torch_branch} ${torch_repo}
                    cd frameworks.ai.pytorch.private-gpu
                    if [[ -n ${torch_commit} ]];then
                        git checkout ${torch_commit}
                    fi
                fi
                cp ${WORKSPACE}/scripts/inductor/inductor_xpu_test.sh ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                cp ${WORKSPACE}/scripts/inductor/inductor_perf_summary.py ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                rm -rf /tmp/torchinductor*
                pushd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                bash inductor_xpu_test.sh ${SUITE} ${DT} ${MODE} ${SCENARIO} xpu 0 static 4 0 & \
                bash inductor_xpu_test.sh ${SUITE} ${DT} ${MODE} ${SCENARIO} xpu 1 static 4 1 & \
                bash inductor_xpu_test.sh ${SUITE} ${DT} ${MODE} ${SCENARIO} xpu 2 static 4 2 & \
                bash inductor_xpu_test.sh ${SUITE} ${DT} ${MODE} ${SCENARIO} xpu 3 static 4 3 & wait
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log ${WORKSPACE}/logs
                popd
                '''
            }catch (Exception e) {
                println('================================================================')
                println('Exception')
                println('================================================================')
                println(e.toString())
            }finally {
                dir("${WORKSPACE}/logs") {
                    archiveArtifacts '**'
                }//dir
            }//finally
        }//if
    }//stage
    stage('Performance-Test') {
        println('================================================================')
        println('Performance-Test')
        println('================================================================')
            if(env.skip_OBO == 'True'){
                sh'''
                set -e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh ${oneapi_ver}
                source ${HOME}/set_proxy.sh
                mkdir -p ${WORKSPACE}/logs
                if [ ! -d ${WORKSPACE}/frameworks.ai.pytorch.private-gpu ];then
                    git clone -b ${torch_branch} ${torch_repo}
                    cd frameworks.ai.pytorch.private-gpu
                    if [[ -n ${torch_commit} ]];then
                        git checkout ${torch_commit}
                    fi
                fi
                cp ${WORKSPACE}/scripts/inductor/inductor_xpu_test.sh ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                cp ${WORKSPACE}/scripts/inductor/inductor_perf_summary.py ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                rm -rf /tmp/torchinductor*
                pushd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                echo -e "========================================================================="
                echo -e "huggingface performance"
                echo -e "========================================================================="
                bash inductor_xpu_test.sh huggingface amp_bf16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh huggingface amp_bf16 training performance xpu 1 & \
                bash inductor_xpu_test.sh huggingface amp_fp16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh huggingface amp_fp16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh huggingface bfloat16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh huggingface bfloat16 training performance xpu 1 & \
                bash inductor_xpu_test.sh huggingface float16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh huggingface float16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh huggingface float32 inference performance xpu 0 & \
                bash inductor_xpu_test.sh huggingface float32 training performance xpu 1 & wait

                echo -e "========================================================================="
                echo -e "timm_models performance"
                echo -e "========================================================================="
                rm -rf /tmp/torchinductor*
                bash inductor_xpu_test.sh timm_models amp_bf16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh timm_models amp_bf16 training performance xpu 1 & \
                bash inductor_xpu_test.sh timm_models amp_fp16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh timm_models amp_fp16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh timm_models bfloat16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh timm_models bfloat16 training performance xpu 1 & \
                bash inductor_xpu_test.sh timm_models float16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh timm_models float16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh timm_models float32 inference performance xpu 0 & \
                bash inductor_xpu_test.sh timm_models float32 training performance xpu 1 & wait

                echo -e "========================================================================="
                echo -e "torchbench performance"
                echo -e "========================================================================="
                rm -rf /tmp/torchinductor*
                pip install tqdm pandas pyre-extensions torchrec tensorboardX dalle2_pytorch torch_geometric scikit-image matplotlib  gym fastNLP doctr matplotlib opacus python-doctr higher opacus dominate kaldi-io librosa effdet pycocotools diffusers
                pip uninstall -y pyarrow pandas
                pip install pyarrow pandas

                git clone https://github.com/facebookresearch/detectron2.git
                python -m pip install -e detectron2

                git clone --recursive https://github.com/facebookresearch/multimodal.git multimodal
                pushd multimodal
                pip install -e .
                popd

                python -m pip uninstall -y torchaudio || true
                wget -q -e use_proxy=no ${torchaudio_whl}
                python -m pip install --force-reinstall $(basename ${torchaudio_whl}) --no-deps
                python -m pip uninstall -y torchvision || true
                wget -q -e use_proxy=no ${torchvision_whl}
                python -m pip install --force-reinstall $(basename ${torchvision_whl}) --no-deps

                git clone --recursive https://github.com/pytorch/text
                pushd text
                python setup.py clean install
                popd

                git clone --recursive https://github.com/pytorch/benchmark.git
                pushd benchmark
                python install.py
                # Note that -e is necessary
                pip install -e .
                popd
                
                bash inductor_xpu_test.sh torchbench amp_bf16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh torchbench amp_bf16 training performance xpu 1 & \
                bash inductor_xpu_test.sh torchbench amp_fp16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh torchbench amp_fp16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh torchbench bfloat16 inference performance xpu 0 & \
                bash inductor_xpu_test.sh torchbench bfloat16 training performance xpu 1 & \
                bash inductor_xpu_test.sh torchbench float16 inference performance xpu 2 & \
                bash inductor_xpu_test.sh torchbench float16 training performance xpu 3 & wait
                bash inductor_xpu_test.sh torchbench float32 inference performance xpu 0 & \
                bash inductor_xpu_test.sh torchbench float32 training performance xpu 1 & wait
                popd
                '''
            }//if
    }//stage
    stage('Perf Test Results Generate and Overview') {
        println('================================================================')
        println('Perf Test Results Generate and Overview')
        println('================================================================')
        if(env.skip_OBO == 'True'){
            try{
                sh'''
                set -e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh ${oneapi_ver}

                pip install styleFrame scipy pandas
                
                pushd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                python inductor_perf_summary.py -s huggingface -p amp_bf16 amp_fp16 bfloat16 float16 float32
                python inductor_perf_summary.py -s timm_models -p amp_bf16 amp_fp16 bfloat16 float16 float32
                python inductor_perf_summary.py -s torchbench -p amp_bf16 amp_fp16 bfloat16 float16 float32
                popd
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log ${WORKSPACE}/logs
                '''
            }catch (Exception e) {
                println('================================================================')
                println('Exception')
                println('================================================================')v
                println(e.toString())
            }finally {
                dir("${WORKSPACE}/logs") {
                    archiveArtifacts '**'
                }//dir
            }//finally
        }//if
    }//stage
    stage('Accuracy-Test') {
        println('================================================================')
        println('Accuracy-Test')
        println('================================================================')
        if(env.skip_OBO == 'True'){
            try {
                sh'''
                set +e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh ${oneapi_ver}
                source ${HOME}/set_proxy.sh
                if [ ! -d ${WORKSPACE}/frameworks.ai.pytorch.private-gpu ];then
                    git clone -b ${torch_branch} ${torch_repo}
                    cd frameworks.ai.pytorch.private-gpu
                    if [[ -n ${torch_commit} ]];then
                        git checkout ${torch_commit}
                    fi
                fi
                rm -rf /tmp/torchinductor*
                pushd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu
                echo -e "========================================================================="
                echo -e "huggingface accuracy"
                echo -e "========================================================================="
                bash inductor_xpu_test.sh huggingface amp_bf16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh huggingface amp_bf16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh huggingface amp_fp16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh huggingface amp_fp16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh huggingface bfloat16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh huggingface bfloat16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh huggingface float16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh huggingface float16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh huggingface float32 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh huggingface float32 training accuracy xpu 1 & wait

                echo -e "========================================================================="
                echo -e "timm_models accuracy"
                echo -e "========================================================================="
                rm -rf /tmp/torchinductor*
                bash inductor_xpu_test.sh timm_models amp_bf16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh timm_models amp_bf16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh timm_models amp_fp16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh timm_models amp_fp16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh timm_models bfloat16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh timm_models bfloat16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh timm_models float16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh timm_models float16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh timm_models float32 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh timm_models float32 training accuracy xpu 1 & wait

                echo -e "========================================================================="
                echo -e "torchbench accuracy"
                echo -e "========================================================================="
                rm -rf /tmp/torchinductor*
                bash inductor_xpu_test.sh torchbench amp_bf16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh torchbench amp_bf16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh torchbench amp_fp16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh torchbench amp_fp16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh torchbench bfloat16 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh torchbench bfloat16 training accuracy xpu 1 & \
                bash inductor_xpu_test.sh torchbench float16 inference accuracy xpu 2 & \
                bash inductor_xpu_test.sh torchbench float16 training accuracy xpu 3 & wait
                bash inductor_xpu_test.sh torchbench float32 inference accuracy xpu 0 & \
                bash inductor_xpu_test.sh torchbench float32 training accuracy xpu 1 & wait

                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log ${WORKSPACE}/logs
                popd
                '''
            }catch (Exception e) {
                println('================================================================')
                println('Exception')
                println('================================================================')v
                println(e.toString())
            }finally {
                dir("${WORKSPACE}/logs") {
                    archiveArtifacts '**'
                }//dir
            }//finally
        }//if
    }//stage
    stage('ACC Test Results Overview') {
        println('================================================================')
        println('ACC Test Results Overview')
        println('================================================================')
        if(env.skip_OBO == 'True'){
            try {
                sh'''
                #! bin/bash
                set +e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh ${oneapi_ver}
                source ${HOME}/set_proxy.sh
                
                bash ${WORKSPACE}/scripts/inductor/inductor_accuracy_results_check.sh
                '''
            }catch (Exception e) {
                println('================================================================')
                println('Exception')
                println('================================================================')v
                println(e.toString())
            }finally {
                dir("${WORKSPACE}/logs") {
                    archiveArtifacts '**'
                }//dir
            }//finally
        }//if
    }//stage
}