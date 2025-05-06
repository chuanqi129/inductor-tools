param (
    [string]$log_dir = "C:\inductor_cpu_ut_log",
    [string]$envName = "pt_bd"
)

$env:TORCHINDUCTOR_WINDOWS_TESTS = 1

# Activate the new environment
Write-Output "Activating conda environment: $envName"
conda activate $envName

function Test_inductor {
    param (
        [string]$logDir
    )
    $env:TORCHINDUCTOR_CPP_WRAPPER = 0

    python tools/dynamo/verify_dynamo.py

    pytest -v test/inductor/test_torchinductor.py `
        2>&1 | Tee-Object -FilePath "$logDir\inductor_test_torchinductor.log"

    pytest -v test/inductor/test_torchinductor_opinfo.py `
        2>&1 | Tee-Object -FilePath "$logDir\inductor_test_torchinductor_opinfo.log"

    # Windows doesn't support AOT
    # pytest -v test/inductor/test_aot_inductor.py `
    #     2>&1 | Tee-Object -FilePath "$logDir\inductor_test_aot_inductor.log"
}
function Test_inductor_cpp_wrapper {
    param (
        [string]$logDir
    )
    $env:TORCHINDUCTOR_WINDOWS_TESTS = 1
    $env:TORCHINDUCTOR_CPP_WRAPPER = 1

    pytest -v test/inductor/test_torchinductor_opinfo.py -k 'linalg or to_sparse' `
        2>&1 | Tee-Object -FilePath "$logDir\cpp_test_torchinductor_opinfo.log"

    pytest -v test/inductor/test_torchinductor.py `
        2>&1 | Tee-Object -FilePath "$logDir\cpp_test_torchinductor.log"

    # AssertionError: Torch not compiled with CUDA enabled
    # pytest -v test/inductor/test_max_autotune.py `
    #     2>&1 | Tee-Object -FilePath "$logDir\cpp_test_test_max_autotune.log"

    pytest -v test/inductor/test_cpu_repro.py `
        2>&1 | Tee-Object -FilePath "$logDir\cpp_test_cpu_repro.log"

    pytest -v test/test_torch.py -k 'take' `
        2>&1 | Tee-Object -FilePath "$logDir\cpp_test_torch.log"

}


Write-Output "Running the CPU UT onw windows"
if (Test-Path -Path $log_dir) {
    Remove-Item -Recurse -Force -Path $log_dir
}
New-Item -ItemType Directory -Path $log_dir
Write-Output "The log directory is: $log_dir"

Test_inductor -logDir $log_dir
Test_inductor_cpp_wrapper -logDir $log_dir
