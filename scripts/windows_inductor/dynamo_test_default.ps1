$env:HTTP_PROXY="http://proxy.ims.intel.com:911"
$env:HTTPS_PROXY="http://proxy.ims.intel.com:911"
$env:TORCHINDUCTOR_FREEZING=1
$env:CXX=""
python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\torchbench_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\huggingface_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\huggingface_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\timm_models_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_msvc\timm_models_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency

$env:CXX="icx"
python benchmarks/dynamo/torchbench.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_icc\torchbench_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/torchbench.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_icc\torchbench_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/huggingface.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_icc\huggingface_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/huggingface.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_icc\huggingface_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency

python benchmarks/dynamo/timm_models.py --performance --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_default\output_dir_icc\timm_models_perf_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llava -x simple_gpt_tp_manual -x stable_diffusion -x llama_v2_7b_16h -x hf_clip -x tacotron2 -x nanogpt -x hf_T5_generate -x torchrec_dlrm -x simple_gpt -x hf_Whisper -x stable_diffusion_text_encoder -x sam -x sam_fast -x detectron2_maskrcnn -x cm3leon_generate -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "D:\weizhuoz\pytorch\output_dir_default\output_dir_icc\timm_models_perf_test_output.log"

python benchmarks/dynamo/timm_models.py --accuracy --float32 -dcpu --output=D:\weizhuoz\pytorch\output_dir_default\output_dir_icc\timm_models_accuracy_test.csv --inference -n50 --inductor --timeout 9000 --freezing --no-skip --dashboard -x llama_v2_7b_16h --cold-start-latency 2>&1 | Tee-Object -FilePath "D:\weizhuoz\pytorch\output_dir_default\output_dir_icc\timm_models_accuracy_test_output.log"
