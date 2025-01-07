param (
    [string]$envName = "pt_win",
    [string]$pythonVersion = "3.10",
    [bool]$create_conda_env = $true
)

$env:HTTP_PROXY = "http://proxy.ims.intel.com:911"
$env:HTTPS_PROXY = "http://proxy.ims.intel.com:911"
$env:DISTUTILS_USE_SDK = 1

git clone --depth=1 https://github.com/pytorch/pytorch.git

$envList = conda env list | Select-String -Pattern "^\s*$envName\s"

if ($create_conda_env or !$envList) {
    # Create a new conda environment
    Write-Host "Creating conda environment: $envName with Python $pythonVersion"

    # Check if the conda environment exists and remove it if it does
    if ($envList) {
        Write-Host "Environment $envName exists. Removing it..."
        conda env remove -y -n $envName
    }
    conda create -y -n $envName python=$pythonVersion

    # Activate the new environment
    Write-Host "Activating conda environment: $envName"
    conda activate $envName

    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

    git clone --depth=1 https://github.com/pytorch/benchmark.git
    Set-Location benchmark
    pip install --no-deps -r requirements.txt
    pip install  safetensors portalocker tokenizers==0.19 huggingface_hub regex botocore ninja

    python install.py --continue_on_fail --no-build-isolation
}


