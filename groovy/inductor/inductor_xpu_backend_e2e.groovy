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
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface ${WORKSPACE}/logs
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
            popd
            cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface ${WORKSPACE}/logs
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
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface/amp_bf16/*accuracy* ${WORKSPACE}/logs
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface/amp_fp16/*accuracy* ${WORKSPACE}/logs
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface/bfloat16/*accuracy* ${WORKSPACE}/logs
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface/float16/*accuracy* ${WORKSPACE}/logs
                cp -r ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface/float32/*accuracy* ${WORKSPACE}/logs
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

                cd ${WORKSPACE}/frameworks.ai.pytorch.private-gpu/inductor_log/huggingface
                cd amp_bf16
                echo -e "============ Acc Check for HF amp_bf16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                csv_lines_inf=$(cat inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
                let num_total_amp_bf16=csv_lines_inf-1
                num_passed_amp_bf16_inf=$(grep "pass" inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_amp_bf16_inf=num_total_amp_bf16-num_passed_amp_bf16_inf
                amp_bf16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_inf'/'$num_total_amp_bf16')*100}'`
                echo "num_total_amp_bf16: $num_total_amp_bf16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_amp_bf16_inf: $num_passed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_bf16_inf: $num_failed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_bf16_inf_acc_pass_rate: $amp_bf16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                num_passed_amp_bf16_tra=$(grep "pass" inductor_huggingface_amp_bf16_training_xpu_accuracy.csv | wc -l)
                let num_failed_amp_bf16_tra=num_total_amp_bf16-num_passed_amp_bf16_tra
                amp_bf16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_bf16_tra'/'$num_total_amp_bf16')*100}'`
                echo "num_passed_amp_bf16_tra: $num_passed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_bf16_tra: $num_failed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_bf16_tra_acc_pass_rate: $amp_bf16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                cd ../amp_fp16
                echo -e "============ Acc Check for HF amp_fp16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                csv_lines_inf=$(cat inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
                let num_total_amp_fp16=csv_lines_inf-1
                num_passed_amp_fp16_inf=$(grep "pass" inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_amp_fp16_inf=num_total_amp_fp16-num_passed_amp_fp16_inf
                amp_fp16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_inf'/'$num_total_amp_fp16')*100}'`
                echo "num_total_amp_fp16: $num_total_amp_fp16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_amp_fp16_inf: $num_passed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_fp16_inf: $num_failed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_fp16_inf_acc_pass_rate: $amp_fp16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                num_passed_amp_fp16_tra=$(grep "pass" inductor_huggingface_amp_fp16_training_xpu_accuracy.csv | wc -l)
                let num_failed_amp_fp16_tra=num_total_amp_fp16-num_passed_amp_fp16_tra
                amp_fp16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_amp_fp16_tra'/'$num_total_amp_fp16')*100}'`
                echo "num_passed_amp_fp16_tra: $num_passed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_fp16_tra: $num_failed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_fp16_tra_acc_pass_rate: $amp_fp16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                cd ../bfloat16
                echo -e "============ Acc Check for HF bfloat16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
                csv_lines_inf=$(cat inductor_huggingface_bfloat16_inference_xpu_accuracy.csv | wc -l)
                let num_total_bfloat16=csv_lines_inf-1
                num_passed_bfloat16_inf=$(grep "pass" inductor_huggingface_bfloat16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_bfloat16_inf=num_total_bfloat16-num_passed_bfloat16_inf
                bfloat16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_inf'/'$num_total_bfloat16')*100}'`
                echo "num_total_bfloat16: $num_total_bfloat16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_bfloat16_inf: $num_passed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_bfloat16_inf: $num_failed_bfloat16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "bfloat16_inf_acc_pass_rate: $bfloat16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                num_passed_bfloat16_tra=$(grep "pass" inductor_huggingface_bfloat16_training_xpu_accuracy.csv | wc -l)
                let num_failed_bfloat16_tra=num_total_bfloat16-num_passed_bfloat16_tra
                bfloat16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_bfloat16_tra'/'$num_total_bfloat16')*100}'`
                echo "num_passed_bfloat16_tra: $num_passed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_bfloat16_tra: $num_failed_bfloat16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "bfloat16_tra_acc_pass_rate: $bfloat16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                cd ../float16
                echo -e "============ Acc Check for HF float16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
                csv_lines_inf=$(cat inductor_huggingface_float16_inference_xpu_accuracy.csv | wc -l)
                let num_total_float16=csv_lines_inf-1
                num_passed_float16_inf=$(grep "pass" inductor_huggingface_float16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_float16_inf=num_total_float16-num_passed_float16_inf
                float16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_inf'/'$num_total_float16')*100}'`
                echo "num_total_float16: $num_total_float16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_float16_inf: $num_passed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_float16_inf: $num_failed_float16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "float16_inf_acc_pass_rate: $float16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                num_passed_float16_tra=$(grep "pass" inductor_huggingface_float16_training_xpu_accuracy.csv | wc -l)
                let num_failed_float16_tra=num_total_float16-num_passed_float16_tra
                float16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float16_tra'/'$num_total_float16')*100}'`
                echo "num_passed_float16_tra: $num_passed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_float16_tra: $num_failed_float16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "float16_tra_acc_pass_rate: $float16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                cd ../float32
                echo -e "============ Acc Check for HF float32 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log        
                csv_lines_inf=$(cat inductor_huggingface_float32_inference_xpu_accuracy.csv | wc -l)
                let num_total_float32=csv_lines_inf-1
                num_passed_float32_inf=$(grep "pass" inductor_huggingface_float32_inference_xpu_accuracy.csv | wc -l)
                let num_failed_float32_inf=num_total_float32-num_passed_float32_inf
                float32_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_inf'/'$num_total_float32')*100}'`
                echo "num_total_float32: $num_total_float32" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_float32_inf: $num_passed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_float32_inf: $num_failed_float32_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "float32_inf_acc_pass_rate: $float32_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                num_passed_float32_tra=$(grep "pass" inductor_huggingface_float32_training_xpu_accuracy.csv | wc -l)
                let num_failed_float32_tra=num_total_float32-nunum_passed_float32_tra
                float32_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%",('$num_passed_float32_tra'/'$num_total_float32')*100}'`
                echo "num_passed_float32_tra: $num_passed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_float32_tra: $num_failed_float32_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "float32_tra_acc_pass_rate: $float32_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
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