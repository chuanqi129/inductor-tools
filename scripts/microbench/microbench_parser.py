import argparse
import os
import pandas as pd
from pandas import ExcelWriter

parser = argparse.ArgumentParser(description='PyTorch models inference')
parser.add_argument('-w', '--workday', type=str, help='workday of refresh')
args = parser.parse_args()

path=os.getcwd()
files = os.listdir(path)
torchbench=""
huggingface=""
timm=""
for file in files:
    if file.startswith("multi_threads_opbench_torchbench"):
        torchbench=file
    elif file.startswith('multi_threads_opbench_huggingface'):
        huggingface=file
    elif file.startswith('multi_threads_opbench_timm'):
        timm=file

def report_generate(file):
    success_ops = {}
    error_ops = {}
    skipped_ops = {}
    with open(file) as reader:
        lines = reader.readlines()
        op = ""
        for line in lines:
            if line.strip() == "":
                continue
            if line.startswith("Running"):
                op = line.split(" ")[1].strip()
            # Skipping aten.convolution_backward.default, no inductor impl
            elif line.startswith("Skipping"):
                op = line.split(",")[0].split(" ")[1].strip()
                reason = line.split(",")[1].strip()
                if op not in skipped_ops.keys():
                    skipped_ops[op] = reason
                    # print(op+", " + reason)
            # error aten.stack.default
            elif line.startswith("error"):
                op = line.split(" ")[1].strip()
                if op not in error_ops.keys():
                    error_ops[op] = "error"
                    # print(op + ", error")
            if line.startswith("Inductor"):
                speedups = line.split("[")[-1].split("]")[0]
                if op not in success_ops.keys():
                    success_ops[op] = speedups
                    # print(op+", "+speedups)
    results = []
    for op in sorted(success_ops):
        results.append(op + ", " + success_ops[op]+ "\n")
    for op in sorted(skipped_ops):
        results.append(op + ", " + skipped_ops[op]+ "\n")
    for op in sorted(error_ops):
        results.append(op + ", " + error_ops[op]+ "\n")
    
    r=pd.DataFrame(results)
    data=pd.DataFrame(r[0].str.split(", ",expand=True))
    data.sort_values(by=[3],inplace=True)
    return data

header=["op_name", "speedup_0.2", "speedup_0.5", "speedup_0.8"]
h = pd.DataFrame(columns=header)
with ExcelWriter(args.workday+'.xlsx') as writer:
    for file in torchbench,huggingface,timm:
        if os.path.exists(file):         
            h.to_excel(writer, sheet_name=str(file.split("_")[3]), index=False,startrow=0, startcol=0)
            report_generate(file).to_excel(writer, sheet_name=str(file.split("_")[3]), index=False,header=False,startrow=1, startcol=0)
