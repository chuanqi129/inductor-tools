installed_torch_git_version=$(python -c "import torch;print(torch.version.git_version)"|| true)
echo -e "[ INFO ] Installed Torch Hash: $installed_torch_git_version"
current_torch_git_version=${torch_commit}
echo -e "[ INFO ] Current Torch Hash: $current_torch_git_version"
if [[ -z "$(pip list | grep torch)" || "$installed_torch_git_version" != "$current_torch_git_version" ]];then
    echo -e "========================================================================="
    echo "Public torch BUILD"
    echo -e "========================================================================="
    pip uninstall torch -y
    git clone -b ${torch_branch} ${torch_repo}
    pushd pytorch || exit 1
    git checkout ${torch_commit}
    git submodule sync
    git submodule update --init --recursive --jobs 0
    conda install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses mkl mkl-include
    python setup.py bdist_wheel 2>&1 | tee pytorch_build.log
    pip install dist/*.whl
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "[ERROR] Public-torch BUILD FAIL"
        exit 1
    fi
    popd
else
    echo -e "========================================================================="
    echo "Public-torch READY"
    echo -e "========================================================================="
fi

source ${HOME}/env.sh
installed_IPEX_git_version=$(python -c "import torch, intel_extension_for_pytorch;print(intel_extension_for_pytorch.__ipex_gitrev__)"|| true)
echo -e "[ INFO ] Installed IPEX Hash: $installed_IPEX_git_version"
current_IPEX_git_version=${ipex_commit}
current_IPEX_version=${current_IPEX_git_version: 0: 9}
echo -e "[ INFO ] Current IPEX Hash: $current_IPEX_version"
if [[ -z "$(pip list | grep intel-extension-for-pytorch)" || "$installed_IPEX_git_version" != "$current_IPEX_version" ]];then
    echo -e "========================================================================="
    echo "IPEX BUILD"
    echo -e "========================================================================="
    pip uninstall intel_extension_for_pytorch -y
    git clone -b ${ipex_branch} ${ipex_repo}
    pushd intel-extension-for-pytorch || exit 1
    git checkout ${ipex_commit}
    git submodule sync
    git submodule update --init --recursive --jobs 0
    pip install -r requirements.txt
    python setup.py bdist_wheel 2>&1 | tee ipex_build.log
    pip install dist/*.whl
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "[ERROR] IPEX BUILD FAIL"
        exit 1
    fi
    popd
else
    echo -e "========================================================================="
    echo "IPEX READY"
    echo -e "========================================================================="
fi
