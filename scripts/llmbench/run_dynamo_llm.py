"""run_dynamo_llm.py
llm inductor inference latency benchmark
Usage:
  python run_dynamo_llm.py --use_dynamo --token_latency --cpp_wrapper --precision float32
"""

#import intel_extension_for_pytorch as ipex
import time
import argparse
import logging
import torch
import torch._dynamo as dynamo
#torch._dynamo.config.verbose=True
#torch._dynamo.config.log_level='DEBUG'
torch._dynamo.config.suppress_errors = True
# args
import torch._inductor.config as config
config.cpp.enable_kernel_profile=True
config.profiler_mark_wrapper_call=True
#config.cpp_wrapper = True
config.freezing = True
#import itt
#itt.pause()
torch._dynamo.config.verbose=True
# _dynamo.config changed https://github.com/pytorch/pytorch/pull/99224
torch._logging.set_logs(dynamo=logging.DEBUG)
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument('--precision', default='bfloat16', type=str, help="float32 or bfloat16")
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--greedy', action='store_true')
# parser.add_argument('--use_ipex_optimize_api', action='store_true')
parser.add_argument('--use_dynamo', action='store_true')
parser.add_argument('--cpp_wrapper', action='store_true')
parser.add_argument("--token_latency", action="store_true")
parser.add_argument('--profile', action='store_true')


args = parser.parse_args()
print(args)

amp_enabled = True if args.precision != "float32" else False
amp_dtype = torch.bfloat16 if args.precision != "float32" else torch.float32

if args.cpp_wrapper:
    config.cpp_wrapper = True
else:
    config.cpp_wrapper = False

def run_benchmark(model):
    # load model
    if model == "gptj6B":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "/workspace/huggingface/gpt-j/gptj6B"
        model = AutoModelForCausalLM.from_pretrained(model_id,low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model == "llama7B":
        from transformers import LlamaForCausalLM, LlamaTokenizer
        model_id = "/workspace/huggingface/llama-7B"
        model = LlamaForCausalLM.from_pretrained(model_id,low_cpu_mem_usage=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_id)    
    model = model.eval()
    # to channels last
    model = model.to(memory_format=torch.channels_last)
    model = model.to(amp_dtype)
    if args.use_dynamo:
        #explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = torch._dynamo.explain(model.generate)
        model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)
        #model.transformer=torch.compile(model.transformer)
    # to ipex
    # if args.use_ipex_optimize_api:
    #     if args.use_dynamo:
    #         assert(False, "ipex.optimize can be applied to the dynamo optimized model")
    #     model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

    # input prompt
    # prompt = "Once upon a time,"
    # 32 tokens input
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
            " She wanted to go to places and meet new people, and have fun."

    # generate args
    if args.greedy:
        generate_kwargs = dict(do_sample=False, temperature=0.9)
    else:
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
    if args.token_latency:
        generate_kwargs["token_latency"] = True
    # start
    total_time = 0.0
    num_iter = 5
    num_warmup = 3
    total_list = []
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1), flush=True)
        prof.export_chrome_trace("my_trace.log" + str(prof.step_num) + ".json")
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=4,
                active=1),
            on_trace_ready=trace_handler
            ) as prof:
        #with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            for i in range(num_iter):
                tic = time.time()
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(gen_ids,skip_special_tokens=True)
                toc = time.time()
                if args.profile:
                    prof.step()
                print(gen_text, flush=True)
                if i >= num_warmup:
                    total_time += (toc - tic)
                    if args.token_latency:
                        total_list.append(output[1])
    print("Inference latency: %.3f s." % (total_time / (num_iter - num_warmup)))
    # token_latency
    if args.token_latency:
        import numpy as np
        from itertools import chain
        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        p90_latency = average_2n[int(len(average_2n) * 0.9)]
        p99_latency = average_2n[int(len(average_2n) * 0.99)]
        print("First token average latency: %.3f sec." % first_latency)
        print("Average 2... latency: %.3f sec." % average_2n_latency)
        print("P90 2... latency: %.3f sec." % p90_latency)
        print("P99 2... latency: %.3f sec." % p99_latency)

if __name__ == '__main__':
    for model in "gptj6B","llama7B":
        run_benchmark(model)
