# Activate the environment
Write-Output "Activating conda environment pt_nightly"
conda activate pt_nightly
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

$line = (pip list | Select-String "torch ").Line
$pt_nightly = [regex]::Match($line, "dev(\d+)").Groups[1].Value
Write-Host $pt_nightly

Set-Location C:\pytorch
git checkout nightly
git pull

if (Test-Path ".\cpu_ut.ps1") {
    Write-Host "Running cpu_ut.ps1..."
    & .\cpu_ut.ps1 -log_dir "C:\inductor_cpu_ut_log_$pt_nightly"
    Write-Host "cpu_ut.ps1 completed"
} else {
    Write-Host "Error: cpu_ut.ps1 not found" -ForegroundColor Red
}

