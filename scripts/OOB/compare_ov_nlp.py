import copy
import openvino as ov
import torch
import time
import numpy as np
from openvino import Type
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Example task
import gc # Garbage Collector
import os


# --- Define Hugging Face models (replace with models you want to test) ---
model_names = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-uncased",                                     # Foundational and widely used
    "roberta-base",                                          # Robust and high-performing
    "distilbert-base-uncased",                              # Smaller, faster alternative to BERT
    "nlptown/bert-base-multilingual-uncased-sentiment",     # Popular for multilingual sentiment
    "cardiffnlp/twitter-roberta-base-sentiment-latest",    # Widely used for Twitter sentiment
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english", # Popular for sentiment analysis
    "j-hartmann/emotion-english-distilroberta-base",        # Commonly used for emotion classification
    "finiteautomata/bertweet-base-sentiment-analysis",      # Popular for Twitter sentiment analysis
    "unitary/toxic-bert",                                    # Widely used for toxicity detection
    "siebert/sentiment-roberta-large-english",
    # Add more models as needed
]
# Define the task for AutoModel loading
hf_task = "sequence-classification"

# --- Benchmark Settings ---
num_iterations = 200
warmup_iterations = 50
batch_size = 16        # Adjust based on your XPU memory
sequence_length = 128
openvino_device = 'GPU' # Target OV device (can be CPU, GPU, AUTO etc.)
torch_device = 'xpu' if torch.xpu.is_available() else 'cpu'
benchmark_dtype = torch.float32 # Use torch.float32 or torch.bfloat16 (check XPU support for bf16)

# --- Function to prepare Hugging Face input data ---
def prepare_hf_inputs(tokenizer, batch_size, sequence_length, device):
    """Generates dummy input data for HF models and moves to target device."""
    dummy_text = ["This is a sample sentence for benchmarking."] * batch_size
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt" # Return PyTorch tensors
    )
    # Move inputs to the target device for torch benchmarking
    return {k: v.to(device) for k, v in inputs.items()}

# --- Function to benchmark a PyTorch model with torch.compile ---
def benchmark_torch_compile(model, input_dict, benchmark_iterations, warmup_iterations, device, dtype=torch.float32):
    
    if not hasattr(torch, 'compile'):
        print("torch.compile() is not available in this PyTorch version.")
        return None

    model_xpu = copy.deepcopy(model).to(dtype).to(device)
    input_dict_xpu = {k: v.to(device) for k,v in input_dict.items()}
    torch._inductor.config.freezing = True
    torch._inductor.config.force_disable_caches = True
    compiled_model = torch.compile(model_xpu, backend="inductor")
    
    print(f"\n--- Benchmarking torch.compile ({benchmark_iterations} iterations, {device}, dtype={dtype}) ---")
    times = []

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup_iterations):
            _ = compiled_model(**input_dict_xpu)

        # Benchmark
        if device == 'xpu':
            torch.xpu.synchronize() # Sync before starting timer

        start_time = time.perf_counter()
        for i in range(benchmark_iterations):
            iter_start_time = time.perf_counter()
            _ = compiled_model(**input_dict_xpu) # Use dictionary unpacking
            if device == 'xpu':
                torch.xpu.synchronize() # Native torch sync after each iteration
            iter_end_time = time.perf_counter()
            times.append(iter_end_time - iter_start_time)
        # No end sync needed if last op was sync
        end_time = time.perf_counter()

        # --- Calculate Metrics ---
        batch_dim = input_dict_xpu['input_ids'].shape[0] # Get batch size from input
        total_time = end_time - start_time

        average_latency_ms = np.mean(times) * 1000
        median_latency_ms = np.median(times) * 1000
        total_items = batch_dim * len(times)
        fps = total_items / total_time

        print(f"Input Batch Size: {batch_dim}")
        print(f"Total inference time (torch.compile): {total_time:.3f} seconds")
        print(f"Average latency (torch.compile): {average_latency_ms:.3f} ms")
        print(f"Median latency (torch.compile): {median_latency_ms:.3f} ms")
        print(f"Throughput (FPS) (torch.compile): {fps:.2f}")
        print("-" * 20)
        return compiled_model

    del model_target_device
    del compiled_model
    if device == 'xpu':
        torch.xpu.empty_cache() # Native torch cache clear
    gc.collect()


# --- Function to benchmark an OpenVINO model ---
def benchmark_openvino(ireq, benchmark_iterations, warmup_iterations, device_name):
    """Benchmarks OpenVINO inference using a pre-filled inference request."""
    inference_times = []
    print(f"\n--- Benchmarking OpenVINO ({benchmark_iterations} iterations) on {device_name} ---")

    # Warm-up
    for _ in range(warmup_iterations):
        ireq.infer()

    # Inference loop
    start_time = time.perf_counter()
    for i in range(benchmark_iterations):
        iter_start_time = time.perf_counter()
        ireq.infer() # Execute inference with pre-loaded data
        iter_end_time = time.perf_counter()
        inference_times.append(iter_end_time - iter_start_time)
    end_time = time.perf_counter()

    # --- Calculate Metrics ---
    total_time = end_time - start_time
    try:
        input_tensor = ireq.get_input_tensor(0)
        batch_dim = input_tensor.shape[0]
    except IndexError:
        print("Warning: Could not determine batch size from OV input tensor, assuming 1.")
        batch_dim = 1 # Fallback

    average_latency_ms = np.mean(inference_times) * 1000
    median_latency_ms = np.median(inference_times) * 1000
    total_items = batch_dim * benchmark_iterations
    fps = total_items / total_time

    print(f"Input Batch Size: {batch_dim}")
    print(f"Total inference time (OpenVINO): {total_time:.3f} seconds")
    print(f"Average latency (OpenVINO): {average_latency_ms:.3f} ms")
    print(f"Median latency (OpenVINO): {median_latency_ms:.3f} ms")
    print(f"Throughput (FPS) (OpenVINO): {fps:.2f}")
    print("-" * 20)

# --- Main benchmarking loop ---
def run_benchmarks(model_names, hf_task, batch_size, sequence_length, torch_device_to_use, openvino_device_target, num_iterations, warmup_iterations, dtype):

    core = ov.Core() # Initialize OpenVINO Core once

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Benchmarking Model: {model_name} (Task: {hf_task})")
        print(f"Batch Size: {batch_size}, Sequence Length: {sequence_length}, Dtype: {dtype}")
        print(f"Torch Device: {torch_device_to_use}, OpenVINO Device: {openvino_device_target}")
        print(f"{'='*50}")

        model_hf = None
        tokenizer = None
        ov_model = None
        compiled_ov_model = None
        ireq = None
        compiled_torch_model = None
        example_input_dict_torch_device = None # For torch benchmark
        example_input_dict_torch_cpu = None # For OV conversion

        try:
            # --- Load Hugging Face Model and Tokenizer ---
            print(f"Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Loading model {model_name} for {hf_task}...")
            model_hf = AutoModelForSequenceClassification.from_pretrained(model_name)
            model_hf.eval() # Set model to evaluation mode

            # --- Prepare Example Input for BOTH Torch and OV ---
            example_input_dict = prepare_hf_inputs(tokenizer, batch_size, sequence_length, device="cpu")
            for k, v in example_input_dict.items():
                print(k)
                print(v.shape)

            # --- Benchmark torch.compile  ---
            compiled_torch_model = benchmark_torch_compile(
                model_hf, # Pass the original CPU model (it will be copied and moved inside)
                example_input_dict,
                num_iterations,
                warmup_iterations,
                torch_device_to_use, # Should be 'xpu'
                dtype=dtype
            )
            # Small cleanup after torch.compile benchmark run
            del compiled_torch_model
            if torch_device_to_use == 'xpu':
                torch.xpu.empty_cache()
            gc.collect()


            # --- Convert to OpenVINO ---
            print(f"\nConverting {model_name} to OpenVINO IR...")
            # Use the CPU version of the model and inputs for conversion
            ov_model = ov.convert_model(model_hf, example_input=example_input_dict, input={k: v.shape for k, v in example_input_dict.items()})

            # --- Compile OpenVINO Model ---
            compile_config = {"PERFORMANCE_HINT": "THROUGHPUT"} # Or LATENCY

            ov_dtype_str = "FP32" # Default
            if dtype == torch.float16:
                 ov_dtype_str = "FP16"
            elif dtype == torch.bfloat16:
                 ov_dtype_str = "BF16"

            if openvino_device_target != 'CPU':
                compile_config["INFERENCE_PRECISION_HINT"] = ov_dtype_str

            compiled_ov_model = core.compile_model(ov_model, openvino_device_target, compile_config)

            # --- Create Inference Request and Prepare Input Data ---
            print("Creating OV inference request and setting input data...")
            ireq = compiled_ov_model.create_infer_request()
            example_input_dict_np = {k: v.numpy() for k, v in example_input_dict.items()}

            for model_input in compiled_ov_model.inputs:
                input_tensor = ireq.get_tensor(model_input)
                input_name = model_input.any_name
                if input_name in example_input_dict_np:
                    if list(input_tensor.shape) == list(example_input_dict_np[input_name].shape):
                         input_tensor.data[:] = example_input_dict_np[input_name]
                    else:
                         raise RuntimeError(f"Shape mismatch for OV input '{input_name}': Model={input_tensor.shape}, Got={example_input_dict_np[input_name].shape}")
                else:
                    pass
                     #raise RuntimeError(f"OV Input '{input_name}' not found in example data keys: {list(example_input_dict_np.keys())}")

            # --- Benchmark OpenVINO ---
            benchmark_openvino(
                ireq, num_iterations, warmup_iterations, openvino_device_target
            )

        except Exception as e:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            continue # Move to the next model

        finally:
            # --- Cleanup ---
            print(f"Cleaning up resources for {model_name}...")
            del model_hf
            del tokenizer
            del ov_model
            del compiled_ov_model
            del ireq
            del example_input_dict_torch_device
            del example_input_dict_torch_cpu
            if torch_device_to_use == 'xpu':
                torch.xpu.empty_cache()
            gc.collect()

    print("\n--- Hugging Face Benchmarking Complete ---")

if __name__ == "__main__":
    benchmark_dtype = torch.float16
    run_benchmarks(
        model_names,
        hf_task,
        batch_size,
        sequence_length,
        torch_device,       # Determined automatically above using native torch.xpu
        openvino_device,    # Set manually in config section
        num_iterations,
        warmup_iterations,
        benchmark_dtype     # Set based on bf16 check or manually
    )