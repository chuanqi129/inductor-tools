param (
    [string]$dir = "C:\logs",
    [string]$envName = "pt_win"
)

Write-Output "The log directory is: $dir"

$dirs = @($dir, "$dir\cpp_msvc", "$dir\cpp_icc")

foreach ($d in $dirs) {
    if (!(Test-Path -Path $d)) {
        New-Item -ItemType Directory -Path $d
    }
}

# Activate the new environment
Write-Host "Activating conda environment: $envName"
conda activate $envName

$env:HTTP_PROXY = "http://proxy.ims.intel.com:911"
$env:HTTPS_PROXY = "http://proxy.ims.intel.com:911"
$env:TORCHINDUCTOR_FREEZING = 1
$env:TORCHINDUCTOR_CPP_WRAPPER = 1
$env:CXX = ""

python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\cpp_msvc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\torchbench_perf_test_output.log"

python benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu --output=$dir\cpp_msvc\torchbench_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\torchbench_accuracy_test_output.log"

python benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --output=$dir\cpp_msvc\huggingface_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\huggingface_perf_test_output.log"

python benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --output=$dir\cpp_msvc\huggingface_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\huggingface_accuracy_test_output.log"

python benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --output=$dir\cpp_msvc\timm_models_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\timm_models_perf_test_output.log"

python benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --output=$dir\cpp_msvc\timm_models_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\timm_models_accuracy_test_output.log"

$env:CXX = "icx-cl"
python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\cpp_icc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\torchbench_perf_test_output.log"

python benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu --output=$dir\cpp_icc\torchbench_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\torchbench_accuracy_test_output.log"

python benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --output=$dir\cpp_icc\huggingface_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\huggingface_perf_test_output.log"

python benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --output=$dir\cpp_icc\huggingface_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\huggingface_accuracy_test_output.log"

python benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --output=$dir\cpp_icc\timm_models_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\timm_models_perf_test_output.log"

python benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --output=$dir\cpp_icc\timm_models_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\timm_models_accuracy_test_output.log"
