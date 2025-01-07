param (
    [string]$dir = "C:\logs",
    [string]$envName = "pt_win",
)

Write-Output "The log directory is: $dir"

$dirs = @($dir, "$dir\default_msvc", "$dir\default_icc", "$dir\cpp_msvc", "$dir\cpp_icc")

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
$env:CXX = ""

python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\default_msvc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --only BERT_pytorch  --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\default_msvc\torchbench_perf_test_output.log"


$env:CXX = "icx-cl"
python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\default_icc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --only BERT_pytorch --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\default_icc\torchbench_perf_test_output.log"

$env:TORCHINDUCTOR_FREEZING = 1
$env:TORCHINDUCTOR_CPP_WRAPPER = 1
$env:CXX = ""

python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\cpp_msvc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --only BERT_pytorch --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_msvc\torchbench_perf_test_output.log"


$env:CXX = "icx-cl"
python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=$dir\cpp_icc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --only BERT_pytorch --cold-start-latency 2>&1 | Tee-Object -FilePath "$dir\cpp_icc\torchbench_perf_test_output.log"

