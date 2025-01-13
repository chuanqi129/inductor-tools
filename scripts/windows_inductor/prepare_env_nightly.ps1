param (
    [string]$envName = "pt_win",
    [string]$pythonVersion = "3.10"
)

$env:DISTUTILS_USE_SDK = 1

# Create a new conda environment
Write-Output "Creating conda environment: $envName with Python $pythonVersion"

# Check if the conda environment exists and remove it if it does
$envList = conda env list | Select-String -Pattern "^\s*$envName\s"
if ($envList) {
    Write-Output "Environment $envName exists. Removing it..."
    conda env remove -y -n $envName
}
conda create -y -n $envName python=$pythonVersion

# Activate the new environment
Write-Output "Activating conda environment: $envName"
conda activate $envName

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

Set-Location benchmark
pip install --no-deps -r requirements.txt
pip install  safetensors portalocker tokenizers==0.19 huggingface_hub regex botocore ninja

python install.py --continue_on_fail --no-build-isolation
