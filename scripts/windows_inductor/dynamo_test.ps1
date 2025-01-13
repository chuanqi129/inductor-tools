param (
    [string]$dir = "C:\inductor_log",
    [string]$envName = "pt_win",
    [string]$compiler = "msvc",
    [string]$wrapper = "default",
    [string]$suite = "all",
    [string]$precision = "float32"
)

$env:TORCHINDUCTOR_FREEZING = 1

if ($wrapper -eq "cpp") {
    $env:TORCHINDUCTOR_CPP_WRAPPER = 1
}

# Set the compiler
if ($compiler -eq "msvc") {
    $env:CXX = "cl"
}
elseif ($compiler -eq "icc") {
    $env:CXX = "icx-cl"
}

if ($suite -eq "all") {
    $suite_list = @("torchbench", "huggingface", "timm_models")
}
else {
    $suite_list = @($suite)
}

# Activate the new environment
Write-Output "Activating conda environment: $envName"
conda activate $envName

function Test-torchbench {
    param (
        [string]$logDir
    )

    # Torchbench performance test
    python benchmarks/dynamo/torchbench.py `
        --performance --float32 -dcpu `
        --output=$logDir\inductor_torchbench_float32_inference_cpu_performance.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llava -x simple_gpt_tp_manual -x stable_diffusion `
        -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt `
        -x hf_T5_generate -x torchrec_dlrm -x simple_gpt `
        -x hf_Whisper -x stable_diffusion_text_encoder `
        -x sam -x sam_fast -x detectron2_maskrcnn `
        -x cm3leon_generate -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\torchbench_perf_test_output.log"

    # Torchbench accuracy test
    python benchmarks/dynamo/torchbench.py `
        --accuracy --float32 -dcpu `
        --output=$logDir\inductor_torchbench_float32_inference_cpu_accuracy.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llava -x simple_gpt_tp_manual -x stable_diffusion `
        -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt `
        -x hf_T5_generate -x torchrec_dlrm -x simple_gpt `
        -x hf_Whisper -x stable_diffusion_text_encoder `
        -x sam -x sam_fast -x detectron2_maskrcnn `
        -x cm3leon_generate -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\torchbench_accuracy_test_output.log"
}


function Test-huggingface {
    param (
        [string]$logDir
    )

    # Huggingface performance test
    python benchmarks/dynamo/huggingface.py `
        --performance --float32 -dcpu `
        --output=$logDir\inductor_huggingface_float32_inference_cpu_performance.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\huggingface_perf_test_output.log"

    # Huggingface accuracy test
    python benchmarks/dynamo/huggingface.py `
        --accuracy --float32 -dcpu `
        --output=$logDir\inductor_huggingface_float32_inference_cpu_accuracy.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\huggingface_accuracy_test_output.log"

}


function Test-timm_models {
    param (
        [string]$logDir
    )

    # Timm_models performance test
    python benchmarks/dynamo/timm_models.py `
        --performance --float32 -dcpu `
        --output=$logDir\inductor_timm_models_float32_inference_cpu_performance.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llava -x simple_gpt_tp_manual -x stable_diffusion `
        -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt `
        -x hf_T5_generate -x torchrec_dlrm -x simple_gpt `
        -x hf_Whisper -x stable_diffusion_text_encoder `
        -x sam -x sam_fast -x detectron2_maskrcnn `
        -x cm3leon_generate -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\timm_models_perf_test_output.log"

    # Timm_models accuracy test
    python benchmarks/dynamo/timm_models.py `
        --accuracy --float32 -dcpu `
        --output=$logDir\inductor_timm_models_float32_inference_cpu_accuracy.csv `
        --inference -n50 --inductor `
        --timeout 9000 --freezing --dashboard `
        -x llama_v2_7b_16h `
        --cold-start-latency `
        2>&1 | Tee-Object -FilePath "$logDir\timm_models_accuracy_test_output.log"

}

foreach ($s in $suite_list) {
    Write-Output "Running the E2E benchmark suite: $s"
    $log_dir = "$dir\$s\$precision"
    if (Test-Path -Path $log_dir) {
        Remove-Item -Recurse -Force -Path $log_dir
    }
    New-Item -ItemType Directory -Path $log_dir
    Write-Output "The log directory is: $log_dir"
    & "Test-$s" -logDir $log_dir
}