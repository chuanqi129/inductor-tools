import copy
import openvino as ov
import torch
import torchvision
import time
import numpy as np
from openvino import Type

# --- Define a list of common torchvision models ---
model_names = [
    "resnet50",
    "resnet18",
    "vgg16",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "densenet121",
    "inception_v3",
    "shufflenet_v2_x1_0",
    "vit_b_16",
    "convnext_tiny",
    "regnet_y_400mf",
]

# --- Benchmark Settings ---
num_iterations = 200
warmup_iterations = 500
batch_size = 256
input_shape = (batch_size, 3, 224, 224)
openvino_device = 'GPU'  # Or 'CPU' or 'XPU' if supported by your OpenVINO installation
torch_device = 'xpu' if torch.xpu.is_available() else 'cpu'

# --- Function to benchmark a PyTorch model with torch.compile ---
def benchmark_torch_compile(model, input_tensor, benchmark_iterations, warmup_iterations, device, dtype=torch.float32):
    if not hasattr(torch, 'compile'):
        print("torch.compile() is not available in this PyTorch version.")
        return None

    model_xpu = copy.deepcopy(model).to(dtype).to(device)
    torch._inductor.config.freezing = True
    torch._inductor.config.force_disable_caches = True
    compiled_model = torch.compile(model_xpu, backend="inductor")
    times = []
    input_tensor = copy.deepcopy(input_tensor).to(device).to(dtype)

    print(f"\n--- Benchmarking torch.compile ({benchmark_iterations} iterations, {device}, dtype={dtype}) ---")

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup_iterations):
            _ = compiled_model(input_tensor)

        # Benchmark
        if device == 'xpu':
            torch.xpu.synchronize()
        start_time = time.perf_counter()
        for _ in range(benchmark_iterations):
            iter_start_time = time.perf_counter()
            _ = compiled_model(input_tensor)
            if device == 'xpu':
                torch.xpu.synchronize()
            iter_end_time = time.perf_counter()
            times.append(iter_end_time - iter_start_time)
        end_time = time.perf_counter()
    batch_size = input_tensor.shape[0]
    total_time = end_time - start_time
    average_latency = np.mean(times) * 1000
    median_latency = np.median(times) * 1000
    fps = batch_size * benchmark_iterations / total_time

    print(f"Total inference time (torch.compile): {total_time:.2f} seconds")
    print(f"Average latency (torch.compile): {average_latency:.2f} ms")
    print(f"Median latency (torch.compile): {median_latency:.2f} ms")
    print(f"Frames Per Second (FPS) (torch.compile): {fps:.2f}")
    return compiled_model

# --- Function to benchmark an OpenVINO model ---
def benchmark_openvino(ireq,benchmark_iterations, warmup_iterations, device_name):
    inference_times = []
    print(f"\n--- Benchmarking OpenVINO ({benchmark_iterations} iterations) on {device_name} ---")

    # Warm-up
    for _ in range(warmup_iterations):
        ireq.infer()

    # Inference loop
    start_time = time.perf_counter()
    for _ in range(benchmark_iterations):
        iter_start_time = time.perf_counter()
        ireq.infer()
        iter_end_time = time.perf_counter()
        inference_times.append(iter_end_time - iter_start_time)
    end_time = time.perf_counter()

    batch_size = ireq.get_input_tensor().shape[0]
    total_time = end_time - start_time
    average_latency = np.mean(inference_times) * 1000
    median_latency = np.median(inference_times) * 1000
    fps = batch_size * benchmark_iterations / total_time

    print(f"Total inference time (OpenVINO): {total_time:.2f} seconds")
    print(f"Average latency (OpenVINO): {average_latency:.2f} ms")
    print(f"Median latency (OpenVINO): {median_latency:.2f} ms")
    print(f"Frames Per Second (FPS) (OpenVINO): {fps:.2f}")

openvino_to_numpy_types_map = [
    (Type.boolean, bool),
    (Type.boolean, np.bool_),
    (Type.f16, np.float16),
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.i8, np.int8),
    (Type.i16, np.int16),
    (Type.i32, np.int32),
    (Type.i64, np.int64),
    (Type.u8, np.uint8),
    (Type.u16, np.uint16),
    (Type.u32, np.uint32),
    (Type.u64, np.uint64),
    (Type.bf16, np.uint16),
    (Type.string, str),
    (Type.string, np.str_),
    (Type.string, bytes),
    (Type.string, np.bytes_),
]


def get_dtype(openvino_type: Type) -> np.dtype:
    """Return a numpy.dtype for an openvino element type."""
    np_type = next(
        (np_type for (ov_type, np_type) in openvino_to_numpy_types_map if ov_type == openvino_type),
        None,
    )

    if np_type:
        return np.dtype(np_type)


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)

# --- Main benchmarking loop ---
def run_benchmarks(model_names, input_shape, openvino_device, torch_device, num_iterations, warmup_iterations, dtype=torch.float32):
    example_input = torch.randn(*input_shape).to(dtype)
    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"Benchmarking Model: {model_name} (dtype: {dtype})")
        print(f"{'='*40}")

        # --- Load PyTorch Model ---
        try:
            model = torch.hub.load("pytorch/vision", model_name, weights="DEFAULT")
            model.eval()
            model = model.to(dtype) # Move model to the specified dtype
        except Exception as e:
            print(f"Error loading PyTorch model {model_name}: {e}")
            continue

        # --- Benchmark torch.compile ---
        compiled_torch = benchmark_torch_compile(
            model, example_input, num_iterations, warmup_iterations, torch_device, dtype
        )

        # --- Convert and Benchmark OpenVINO ---
        try:
            ov_model = ov.convert_model(model, example_input=(example_input,), input=example_input.shape)
            core = ov.Core()
            compiled_ov_model = core.compile_model(ov_model, openvino_device, config={"INFERENCE_PRECISION_HINT":'FP16', 'PERFORMANCE_HINT': 'THROUGHPUT'})
            ireq = compiled_ov_model.create_infer_request()
            # Fill input data for the ireq
            print(compiled_ov_model.inputs)
            for model_input in compiled_ov_model.inputs:
                fill_tensor_random(ireq.get_tensor(model_input))
            benchmark_openvino(
                ireq, num_iterations, warmup_iterations, openvino_device
            )
        except Exception as e:
            print(f"Error converting and benchmarking OpenVINO model {model_name}: {e}")

    print("\n--- Benchmarking Complete ---")

if __name__ == "__main__":
    # Run benchmarks with default FP32
     run_benchmarks(model_names, input_shape, openvino_device, torch_device, num_iterations, warmup_iterations, dtype=torch.float16)

    # Optional: Run benchmarks with FP16 (if supported by your hardware and frameworks)
    # run_benchmarks(model_names, input_shape, openvino_device, torch_device, num_iterations, warmup_iterations, dtype=torch.float16)
