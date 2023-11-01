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

        # set gpu governor
        if [[ -z "${USER_PASS}" ]];then USER_PASS="${USER_PASSWORD}";fi
        '''
    }//stage
    stage('Install Dependency') {
        println('================================================================')
        println('Install Dependency')
        println('================================================================')
            sh '''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh oneapi_ver 
            python -m pip install numpy --upgrade --no-cache --force-reinstall

            if [[ -n ${torch_whl} ]] && [[ -n ${ipex_whl} ]];then
                python -m pip uninstall -y torch || true
                wget -q -e use_proxy=no ${torch_whl}
                python -m pip install --force-reinstall $(basename ${torch_whl})
                python -m pip uninstall -y intel_extension_for_pytorch || true
                wget -q -e use_proxy=no ${ipex_whl}
                python -m pip install --force-reinstall $(basename ${ipex_whl})
            else
                bash ${WORKSPACE}/inductor-tools/scripts/env_prepare.sh
                ${torch_repo} \
                ${torch_branch} \
                ${torch_commit} \
                ${ipex_repo} \
                ${ipex_branch} \
                ${ipex_commit} 
                source ${HOME}/env.sh oneapi_ver
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
            source ${HOME}/env.sh oneapi_ver 
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
            '''
        }//retry
    }//stage
    stage('Accuracy-Test') {
        println('================================================================')
        println('Accuracy-Test')
        println('================================================================')
        retry(1){
            sh'''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh oneapi_ver

            cp ${WORKSPACE}/inductor-tools/scripts/inductor_xpu_test.sh ${WORKSPACE}/pytorch
            cp ${WORKSPACE}/inductor-tools/scripts/inductor_perf_summary.py ${WORKSPACE}/pytorch
            pip install styleFrame scipy pandas
            pushd ${WORKSPACE}/pytorch
            rm -rf inductor_log
            bash inductor_xpu_test.sh huggingface amp_bf16 inference accuracy xpu 0 & \
            bash inductor_xpu_test.sh huggingface amp_bf16 training accuracy xpu 1 & \
            bash inductor_xpu_test.sh huggingface amp_fp16 inference accuracy xpu 2 & \
            bash inductor_xpu_test.sh huggingface amp_fp16 training accuracy xpu 3 & wait
            popd
            '''
        }//retry
    }//stage
    stage('ACC Test Results Overview') {
        println('================================================================')
        println('ACC Test Results Overview')
        println('================================================================')
            try {
                sh'''
                set -e
                set +x
                source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
                conda activate ${conda_env}
                source ${HOME}/env.sh oneapi_ver 

                cd ${WORKSPACE}/pytorch/inductor_log/huggingface
                cd amp_bf16
                cp *.log ${WORKSPACE}/logs
                echo -e "============ Acc Check for HF amp_bf16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                csv_lines_inf=$(cat inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
                let num_total_amp_bf16=csv_lines_inf-1
                num_passed_amp_bf16_inf=$(grep "pass" inductor_huggingface_amp_bf16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_amp_bf16_inf=num_total_amp_bf16-num_passed_amp_bf16_inf
                amp_bf16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed_amp_bf16_inf'/'$num_total_amp_bf16')*100}'`
                echo "num_total_amp_bf16: $num_total_amp_bf16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_amp_bf16_inf: $num_passed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_bf16_inf: $num_failed_amp_bf16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_bf16_inf_acc_pass_rate: $amp_bf16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                num_passed_amp_bf16_tra=$(grep "pass" inductor_huggingface_amp_bf16_training_xpu_accuracy.csv | wc -l)
                let num_failed_amp_bf16_tra=num_total_amp_bf16-num_passed_amp_bf16_tra
                amp_bf16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed_amp_bf16_tra'/'$num_total_amp_bf16')*100}'`
                echo "num_passed_amp_bf16_tra: $num_passed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_bf16_tra: $num_failed_amp_bf16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_bf16_tra_acc_pass_rate: $amp_bf16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                cd ../amp_fp16
                cp *.log ${WORKSPACE}/logs
                echo -e "============ Acc Check for HF amp_fp16 ============" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                csv_lines_inf=$(cat inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
                let num_total_amp_fp16=csv_lines_inf-1
                num_passed_amp_fp16_inf=$(grep "pass" inductor_huggingface_amp_fp16_inference_xpu_accuracy.csv | wc -l)
                let num_failed_amp_fp16_inf=num_total_amp_fp16-num_passed_amp_fp16_inf
                amp_fp16_inf_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed_amp_fp16_inf'/'$num_total_amp_fp16')*100}'`
                echo "num_total_amp_fp16: $num_total_amp_fp16" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_passed_amp_fp16_inf: $num_passed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_fp16_inf: $num_failed_amp_fp16_inf" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_fp16_inf_acc_pass_rate: $amp_fp16_inf_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log

                num_passed_amp_fp16_tra=$(grep "pass" inductor_huggingface_amp_fp16_training_xpu_accuracy.csv | wc -l)
                let num_failed_amp_fp16_tra=num_total_amp_fp16-num_passed_amp_fp16_tra
                amp_fp16_tra_acc_pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed_amp_fp16_tra'/'$num_total_amp_fp16')*100}'`
                echo "num_passed_amp_fp16_tra: $num_passed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "num_failed_amp_fp16_tra: $num_failed_amp_fp16_tra" | tee -a ${WORKSPACE}/logs/e2e_summary.log
                echo "amp_fp16_tra_acc_pass_rate: $amp_fp16_tra_acc_pass_rate" | tee -a ${WORKSPACE}/logs/e2e_summary.log
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
    }//stage
    stage('Test Results Check') {
        println('================================================================')
        println('Test Results Check')
        println('================================================================')
        retry(1){
            sh'''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh oneapi_ver 

            cd ${WORKSPACE}/pytorch/inductor_log/huggingface
            cd amp_bf16
            num_passed_amp_bf16_inf=$(grep "num_passed_amp_bf16_inf:" e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
            if [ $num_passed_amp_bf16_inf -lt 45 ]; then
                echo -e "[ERROR] Inductor E2E Nightly test for HF amp_bf16 inference passed_num < 45"
                exit 1
            fi
            num_passed_amp_bf16_tra=$(grep "num_passed_amp_bf16_tra:" e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
            if [ $num_passed_amp_bf16_tra -lt 42 ]; then
                echo -e "[ERROR] Inductor E2E Nightly test for HF amp_bf16 training passed_num < 42"
                exit 1
            fi
            cd ../amp_fp16
            num_passed_amp_fp16_inf=$(grep "num_passed_amp_fp16_inf:" e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
            if [ $num_passed_amp_fp16_inf -lt 45 ]; then
                echo -e "[ERROR] Inductor E2E Nightly test for HF amp_fp16 inference passed_num < 45"
                exit 1
            fi
            num_passed_amp_fp16_tra=$(grep "num_passed_amp_fp16_tra:" e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
            if [ $num_passed_amp_fp16_tra -lt 42 ]; then
                echo -e "[ERROR] Inductor E2E Nightly test for HF amp_fp16 training passed_num < 42"
                exit 1
            fi
            '''
        }//retry
    }//stage
    stage('Performance-Test') {
        println('================================================================')
        println('Performance-Test')
        println('================================================================')
            sh'''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh oneapi_ver 

            pushd ${WORKSPACE}/pytorch
            bash inductor_xpu_test.sh huggingface amp_bf16 inference performance xpu 0 & \
            bash inductor_xpu_test.sh huggingface amp_bf16 training performance xpu 1 & \
            bash inductor_xpu_test.sh huggingface amp_fp16 inference performance xpu 2 & \
            bash inductor_xpu_test.sh huggingface amp_fp16 training performance xpu 3 & wait
            popd
            '''
    }//stage
    stage('Perf Test Results Generate and Overview') {
        println('================================================================')
        println('Perf Test Results Generate and Overview')
        println('================================================================')
        try{
            sh'''
            set -e
            set +x
            source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>&1 >> /dev/null
            conda activate ${conda_env}
            source ${HOME}/env.sh oneapi_ver 

            pip install pandas styleframe
            pushd ${WORKSPACE}/pytorch
            python inductor_perf_summary.py -s huggingface -p amp_bf16 amp_fp16
            popd
            cp ${WORKSPACE}/pytorch/inductor_log/huggingface/amp_bf16/*performance*.log ${WORKSPACE}/logs
            cp ${WORKSPACE}/pytorch/inductor_log/huggingface/amp_fp16/*performance*.log ${WORKSPACE}/logs
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
    }//stage
}