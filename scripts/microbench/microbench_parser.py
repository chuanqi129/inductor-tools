"""microbench_parser.py
Generate report.
Usage:
  python microbench_parser.py -o 20221228 -l https://inteltf-jenk.sh.intel.com/job/inductor_dashboard/100/ -n mlp-validate-icx24-ubuntu
  python microbench_parser.py -o 20221228 -l https://inteltf-jenk.sh.intel.com/job/inductor_dashboard/100/ -n mlp-validate-icx24-ubuntu --html_off
"""

import argparse
import os
import subprocess
import pandas as pd
from pandas import ExcelWriter

parser = argparse.ArgumentParser(description='Torchinductor Microbench Log Parser')
parser.add_argument('-p', '--log_path', type=str, help='log path')
parser.add_argument('-o', '--output', type=str, help='string included in reports name')
parser.add_argument('-l', '--url', type=str, help='jenkins build url')
parser.add_argument('-n', '--node', type=str, help='jenkin node lable')
parser.add_argument('--html_off', action='store_true', help='turn off html report generate')
args = parser.parse_args()

report_name = 'op-microbench-'+args.output
path = args.log_path if args.log_path else os.getcwd()
files = os.listdir(path)
torchbench=""
huggingface=""
timm=""
for file in files:
    if file.startswith("multi_threads_opbench_torchbench"):
        torchbench=path+'/'+file
    elif file.startswith('multi_threads_opbench_huggingface'):
        huggingface=path+'/'+file
    elif file.startswith('multi_threads_opbench_timm'):
        timm=path+'/'+file

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
    for i in range(4):
        data[i]= pd.to_numeric(data[i],errors='ignore')
    data.sort_values(by=[3],inplace=True)  
    data=data.style.highlight_between(subset=[3], axis=0, left=0, right=1, inclusive='both', props='color:black;background-color:#FFDEAD')
    return data


for file in torchbench,huggingface,timm:
    if not file.isspace():
        try:
            data=report_generate(file)        
        except:
            raise Exception('==========error occurred in generating report from '+file+'========')

header=["op_name", "speedup_0.2", "speedup_0.5", "speedup_0.8"]
h = pd.DataFrame(columns=header)
with ExcelWriter(report_name+'.xlsx',mode="w") as writer:
    for file in torchbench,huggingface,timm:
        if not file.isspace():
            report_generate(file).to_excel(writer, sheet_name=str(file.split("_")[3]), index=False,header=False,startrow=1, startcol=0)
            h.to_excel(writer, sheet_name=str(file.split("_")[3]), index=False,startrow=0, startcol=0) 
            


commit_list=[]
url_list=[]
for item in ["pytorch","vision","text","audio","data","benchmark"]:
    get_commit_cmd=f"git ls-remote https://github.com/pytorch/{item}.git refs/heads/nightly"+" | awk '{ print $1 }'" if item != "benchmark" \
        else f"git ls-remote https://github.com/pytorch/{item}.git refs/heads/main"+" | awk '{ print $1 }'"
    commit=subprocess.getstatusoutput(get_commit_cmd)[1].strip()
    commit_short=commit[0:7]
    item_url=f'https://github.com/pytorch/{item}/commit/'+commit
    commit_list.append(commit_short)
    url_list.append(item_url)

def AdditionalInfo():
    return f'<p>job info:</p> \
        <ol><table><tr><td>Build url:&nbsp;</td><td>{args.url}</td></tr></table></ol> \
            <p>SW Info:</p><ol><table> \
                <tbody> \
                    <tr><td>docker image:&nbsp;</td><td>ccr-registry.caas.intel.com/pytorch/pt_inductor:nightly</td></tr> \
                    <tr><td>StockPT:&nbsp;</td><td><a href={url_list[0]}> {commit_list[0]}</a></td></tr> \
                    <tr><td>TORCH_VISION:&nbsp;</td><td><a href={url_list[1]}> {commit_list[1]} </a></td></tr> \
                    <tr><td>TORCH_TEXT:&nbsp;</td><td><a href={url_list[2]}> {commit_list[2]} </a></td></tr> \
                    <tr><td>TORCH_AUDIO:&nbsp;</td><td><a href={url_list[3]}> {commit_list[3]} </a></td></tr> \
                    <tr><td>TORCH_DATA:&nbsp;</td><td><a href={url_list[4]}> {commit_list[4]} </a></td></tr> \
                    <tr><td>TORCH_BENCH:&nbsp;</td><td><a href={url_list[5]}> {commit_list[5]} </a></td></tr> \
                </tbody></table></ol> \
            <p>HW info:</p><ol><table> \
                <tbody> \
                    <tr><td>Machine name:&nbsp;</td><td>{args.node}</td></tr> \
                    <tr><td>Manufacturer:&nbsp;</td><td>Intel Corporation</td></tr> \
                    <tr><td>Kernel:</td><td>5.4.0-131-generic</td></tr> \
                    <tr><td>Microcode:</td><td>0xd000375</td></tr> \
                    <tr><td>Installed Memory:</td><td>503GB</td></tr> \
                    <tr><td>OS:</td><td>Ubuntu 18.04.6 LTS</td></tr> \
                    <tr><td>CPU Model:</td><td>Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz</td></tr> \
                    <tr><td>GCC:</td><td>gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0</td></tr> \
                    <tr><td>GLIBC:</td><td>ldd (Ubuntu GLIBC 2.27-3ubuntu1.5) 2.27</td></tr> \
                    <tr><td>Binutils:</td><td>GNU ld (GNU Binutils for Ubuntu) 2.30</td></tr> \
                    <tr><td>Python:</td><td>Python 3.8.3</td></tr> \
                </tbody></table></ol> \
            <p>You can find details from attachments, Thanks</p>'


if not args.html_off:
    result=[]         
    for file in torchbench,huggingface,timm:
        if not file.isspace():
            df = pd.read_excel(report_name+'.xlsx',sheet_name=file.split("_")[3])
            dt=df[df["speedup_0.8"]<1]
            suite_title = {"model suite": file.split("_")[3]}
            result.append(dt)
            result.append(pd.DataFrame(suite_title, index=[0]))
    data=pd.concat(result).fillna('*')
    data.to_html('ops.html',header = True,index = False, justify='center')
    with open("ops.html",mode = "a") as f:
        f.write(AdditionalInfo())
    f.close()
