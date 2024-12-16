param (
    [string]$envName = "pt_win",
    [string]$pythonVersion = "3.10"
)

$env:HTTP_PROXY = "http://proxy.ims.intel.com:911"
$env:HTTPS_PROXY = "http://proxy.ims.intel.com:911"

Set-PSDebug -Trace 1

# Create a new conda environment
Write-Host "Creating conda environment: $envName with Python $pythonVersion"
# Check if the conda environment exists and remove it if it does
$envList = conda env list | Select-String -Pattern "^\s*$envName\s"
if ($envList) {
    Write-Host "Environment $envName exists. Removing it..."
    conda env remove -y -n $envName
}
conda create -y -n $envName python=$pythonVersion

# Activate the new environment
Write-Host "Activating conda environment: $envName"
conda activate $envName

# Install packages with conda
conda install -y `
    astunparse `
    cffi `
    cython `
    cmake>=3.13.0 `
    dataclasses `
    future `
    git-lfs `
    ipython `
    mkl `
    mkl-include `
    ninja `
    numpy `
    requests `
    typing `
    typing_extensions `
    Pillow `
    pkg-config `
    pybind11 `
    pyyaml `
    setuptools `
    openssl `
    weasyprint



$PT_REPO = "https://github.com/pytorch/pytorch"

git config --global http.postBuffer 524288000
git clone --depth=100 ${PT_REPO}
Set-Location pytorch && git fetch --unshallow && git submodule sync && git submodule update --init --recursive
python setup.py develop && Set-Location ..

$env:DISTUTILS_USE_SDK = 1
git clone https://github.com/pytorch/vision.git 
Set-Location vision && git checkout $(Get-Content ..\pytorch\.github\ci_commit_pins\vision.txt)
python setup.py bdist_wheel
$wheelFile = Get-ChildItem -Path "dist" -Filter "*.whl" | Select-Object -First 1
pip install $wheelFile.FullName && Set-Location ..


git clone https://github.com/pytorch/audio.git
Set-Location audio && git checkout $(Get-Content ../pytorch/.github/ci_commit_pins/audio.txt) && git submodule sync && git submodule update --init --recursive
python setup.py bdist_wheel
$wheelFile = Get-ChildItem -Path "dist" -Filter "*.whl" | Select-Object -First 1
pip install $wheelFile.FullName && Set-Location ..


git clone https://github.com/pytorch/benchmark.git
Set-Location benchmark && git checkout $(Get-Content ..//pytorch/.github/ci_commit_pins/torchbench.txt)
(Get-Content requirements.txt) -notmatch 'numpy' | Set-Content requirements.txt # work around numpy version issue

pip install --no-deps -r requirements.txt
pip install --no-cache Jinja2==3.1.2 markupsafe==2.0.1 beartype==0.15.0 mpmath==1.3.0
python install.py --continue_on_fail
Set-Location ..

$TRANSFORMERS_COMMIT = Get-Content pytorch/.ci/docker/ci_commit_pins/huggingface.txt
pip install --force-reinstall git+https://github.com/huggingface/transformers@$TRANSFORMERS_COMMIT

pip install numpy==1.26.4
