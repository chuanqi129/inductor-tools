param (
    [string]$envName = "pt_nightly",
    [string]$pythonVersion = "3.10"
)

# Create a new conda environment
Write-Output "Creating conda environment: $envName with Python $pythonVersion"

# Check if the conda environment exists and remove it if it does
$envList = conda env list | Select-String -SimpleMatch -Pattern " $envName "
if ($envList) {
    Write-Output "Environment $envName exists. Removing it..."
    conda env remove -y -n $envName
}
conda create -y -n $envName python=$pythonVersion

# Activate the new environment
Write-Output "Installing nightly PyTorch packages in environment: $envName"

#Installs nightly CPU builds of PyTorch packages and test dependencies into the specified conda environment.
conda run -n $envName pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
conda run -n $envName pip install pytest expecttest

