"""run_dynamo_gptj.py
gptj inductor inference throughput benchmark
Usage:
  python run_dynamo_gptj.py --transformers_version 4.24.0 --use_dynamo --precision bf16 --greedy
"""

#import intel_extension_for_pytorch as ipex
import time
import subprocess
import argparse
import torch
import torch._dynamo as dynamo
import pandas as pd
#torch._dynamo.config.verbose=True
#torch._dynamo.config.log_level='DEBUG'
torch._dynamo.config.suppress_errors = True
# args
import torch._inductor.config as config
config.cpp.enable_kernel_profile=True
config.profiler_mark_wrapper_call=True
#config.cpp_wrapper = True
#import itt
#itt.pause()
torch._dynamo.config.verbose=True
torch._dynamo.config.log_level='DEBUG'
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument('--precision', default='bfloat16', type=str, help="float32 or bfloat16")
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--greedy', action='store_true')
# parser.add_argument('--use_ipex_optimize_api', action='store_true')
parser.add_argument('--use_dynamo', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--transformers_version',default='4.24.0', type=str, help="transformers version")


args = parser.parse_args()
print(args)

amp_enabled = True if args.precision != "float32" else False
amp_dtype = torch.bfloat16 if args.precision != "float32" else torch.float32

if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# install transformers
subprocess.run(['pip', 'uninstall', 'transformers', '-y'],shell=True)
subprocess.run(f'pip install transformers=={args.transformers_version}',shell=True)
subprocess.run('pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html',shell=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model
model_id = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()
# to channels last
model = model.to(memory_format=torch.channels_last)
model = model.to(torch.bfloat16)
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


# start
total_time = 0.0
num_iter = 5
num_warmup = 3
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
            gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            toc = time.time()
            if args.profile:
                prof.step()
            print(gen_text, flush=True)
            if i >= num_warmup:
                total_time += (toc - tic)
result = ('%.3f' % (total_time / (num_iter - num_warmup) * 1000))

commit_list=[]
url_list=[]
version = pd.read_table('version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
componment = ["benchmark","pytorch","vision","text","audio","data"]
for item in componment:
    sha_short = version.loc[componment.index(item), "commit"][-7:] if item != "benchmark" \
        else version.loc[componment.index(item),"commit"][-8:]
    commit_list.append(sha_short)    
    url_list.append(f"https://github.com/pytorch/{item}/commit/"+sha_short) 

report_content=f'''<!DOCTYPE html> \
<html> \
<head><title>LLM Model Report</title></head> \
<body> \
    <h3> LLM Model(GPT-J) Inductor Benchmark Report </h3> \
    <p>Result:</p> \
    <table border="1"> \
        <tr> \
            <th>precision</th> \
            <th>max-new-tokens</th> \
            <th>greedy</th> \
            <th>use_dynamo</th> \
            <th>throughput</th> \
        </tr> \
        <tr> \
            <td><p style="text-align:center">{args.precision}</p></td> \
            <td><p style="text-align:center">32</p></td> \
            <td><p style="text-align:center">True</p></td> \
            <td><p style="text-align:center">True</p></td> \
            <td><p style="text-align:center">{result} ms</p></td> \                                   
        </tr> \
    </table> \
    <table border="1"> \
    <p>SW Info:</p> \
        <tr><td>Pytorch:&nbsp;</td><td><a href={url_list[1]}> {commit_list[1]}</a></td></tr> \
        <tr><td>transformers:&nbsp;</td><td>{args.transformers_version}</td></tr> \
        <tr><td>TORCH_VISION:&nbsp;</td><td><a href={url_list[2]}> {commit_list[2]} </a></td></tr> \
        <tr><td>TORCH_TEXT:&nbsp;</td><td><a href={url_list[3]}> {commit_list[3]} </a></td></tr> \
        <tr><td>TORCH_AUDIO:&nbsp;</td><td><a href={url_list[4]}> {commit_list[4]} </a></td></tr> \
        <tr><td>TORCH_DATA:&nbsp;</td><td><a href={url_list[5]}> {commit_list[5]} </a></td></tr> \
        <tr><td>TORCH_BENCH:&nbsp;</td><td><a href={url_list[0]}> {commit_list[0]} </a></td></tr> \
    </table> \
    <h4>Thanks.</h4> \
</body> \
</html> \
'''

with open("llm_report.html",mode = "a") as f:
    f.write(report_content)
f.close()
