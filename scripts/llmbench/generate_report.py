import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Torchinductor LLMbench Report Generate')
parser.add_argument('-l', '--url', type=str, help='jenkins build url')
parser.add_argument('-n', '--node', type=str, default='mlp-validate-icx24-ubuntu',help='benchmark node')
args = parser.parse_args()

commit_list=[]
url_list=[]
result = pd.read_table('result.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
componment = ["benchmark","pytorch","vision","text","audio","data"]
for item in componment:
    sha_short = result.loc[componment.index(item), "commit"][-7:] if item != "benchmark" \
        else result.loc[componment.index(item),"commit"][-8:]
    commit_list.append(sha_short)    
    url_list.append("https://github.com/pytorch/"+item+"/commit/"+sha_short)
precision = result.loc[7,"commit"]
latency = result.loc[8,"commit"]

latency_gptj = latency.split('ms.')[0]
latency_llama = latency.split('ms.')[1]
latency_gptj_cppwrapper = latency.split('ms.')[2]
latency_llama_cppwrapper = latency.split('ms.')[3]

last_latency_gptj=0
last_latency_gptj_cppwrapper=0
last_latency_llama=0
last_latency_llama_cppwrapper=0

try:
    last_result = pd.read_table('llm_bench_'+args.node+'/result.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
    
    last_latency = last_result.loc[8,"commit"]
    last_latency_gptj = last_latency.split('ms.')[0]
    ratio_gptj = float(last_latency_gptj) / float(latency_gptj)
    last_latency_llama = last_latency.split('ms.')[1]
    ratio_llama = float(last_latency_llama) / float(latency_llama)

    last_latency_gptj_cppwrapper = last_latency.split('ms.')[2]
    ratio_gptj_cppwrapper = float(last_latency_gptj_cppwrapper) / float(latency_gptj_cppwrapper)
    last_latency_llama_cppwrapper = last_latency.split('ms.')[3]
    ratio_llama_cppwrapper = float(last_latency_llama_cppwrapper) / float(latency_llama_cppwrapper)    
except:
    last_latency_gptj=0
    last_latency_gptj_cppwrapper=0
    last_latency_llama=0
    last_latency_llama_cppwrapper=0
    ratio_gptj=0
    ratio_llama=0
    ratio_gptj_cppwrapper=0
    ratio_llama_cppwrapper=0
    pass

transformers = result.loc[6,"commit"]

report_content=f'''<!DOCTYPE html>
<html>
<head><title>LLM Model Report</title></head>
<body>
    <h3> LLM Model(GPTJ & LLAMA) Inductor Benchmark Report </h3>
    <p>Result:</p>
    <table border="1">
        <tr>
            <th>Model</th>
            <th>precision</th>
            <th>max-new-tokens</th> 
            <th>greedy</th> 
            <th>use_dynamo</th> 
            <th>cpp_wrapper</th> 
            <th>latency</th> 
            <th>latency(lastsuccessful)</th> 
            <th>ratio(last / current)</th>  
        </tr> 
        <tr> 
            <td><p style="text-align:center">gptj6B</p></td> 
            <td><p style="text-align:center">{precision}</p></td> 
            <td><p style="text-align:center">32</p></td> 
            <td><p style="text-align:center">False</p></td> 
            <td><p style="text-align:center">True</p></td>
            <td><p style="text-align:center">False</p></td>
            <td><p style="text-align:center">{latency_gptj}ms</p></td>                                  
            <td><p style="text-align:center">{last_latency_gptj}ms</p></td>                                 
            <td><p style="text-align:center">{ratio_gptj}</p></td>                                 
        </tr> 
        <tr> 
            <td><p style="text-align:center">gptj6B</p></td> 
            <td><p style="text-align:center">{precision}</p></td> 
            <td><p style="text-align:center">32</p></td> 
            <td><p style="text-align:center">False</p></td> 
            <td><p style="text-align:center">True</p></td> 
            <td><p style="text-align:center">True</p></td>
            <td><p style="text-align:center">{latency_gptj_cppwrapper}ms</p></td>                                  
            <td><p style="text-align:center">{last_latency_gptj_cppwrapper}ms</p></td>                                 
            <td><p style="text-align:center">{ratio_gptj_cppwrapper}</p></td>                                 
        </tr>         
        <tr> 
            <td><p style="text-align:center">llama7B</p></td> 
            <td><p style="text-align:center">{precision}</p></td> 
            <td><p style="text-align:center">32</p></td> 
            <td><p style="text-align:center">False</p></td> 
            <td><p style="text-align:center">True</p></td>
            <td><p style="text-align:center">False</p></td>
            <td><p style="text-align:center">{latency_llama}ms</p></td>                                    
            <td><p style="text-align:center">{last_latency_llama}ms</p></td>                                    
            <td><p style="text-align:center">{ratio_llama}</p></td>                                    
        </tr>
        <tr> 
            <td><p style="text-align:center">llama7B</p></td> 
            <td><p style="text-align:center">{precision}</p></td> 
            <td><p style="text-align:center">32</p></td> 
            <td><p style="text-align:center">False</p></td> 
            <td><p style="text-align:center">True</p></td> 
            <td><p style="text-align:center">True</p></td>
            <td><p style="text-align:center">{latency_llama_cppwrapper}ms</p></td>                                    
            <td><p style="text-align:center">{last_latency_llama_cppwrapper}ms</p></td>                                    
            <td><p style="text-align:center">{ratio_llama_cppwrapper}</p></td>                                    
        </tr> 
    </table> 
    <p>SW Info:</p> 
    <table border="1"> 
        <tr><td>Pytorch:&nbsp;</td><td><a href={url_list[1]}> {commit_list[1]}</a></td></tr> 
        <tr><td>transformers:&nbsp;</td><td>{transformers}</td></tr> 
        <tr><td>TORCH_VISION:&nbsp;</td><td><a href={url_list[2]}> {commit_list[2]} </a></td></tr> 
        <tr><td>TORCH_TEXT:&nbsp;</td><td><a href={url_list[3]}> {commit_list[3]} </a></td></tr> 
        <tr><td>TORCH_AUDIO:&nbsp;</td><td><a href={url_list[4]}> {commit_list[4]} </a></td></tr> 
        <tr><td>TORCH_DATA:&nbsp;</td><td><a href={url_list[5]}> {commit_list[5]} </a></td></tr> 
        <tr><td>TORCH_BENCH:&nbsp;</td><td><a href={url_list[0]}> {commit_list[0]} </a></td></tr> 
    </table> 
    <p>HW info:</p><ol><table> 
        <tbody> 
            <tr><td>Machine name:&nbsp;</td><td>{args.node}</td></tr> 
        </tbody></table></ol> 
    <p>job info:</p><ol><table>
        <tbody>
            <tr><td>Build url:&nbsp;</td><td>{args.url}</td></tr>
        </tbody></table></ol>         
    <h4>Thanks.</h4> 
</body> 
</html> 
'''

with open("llm_report.html",mode = "a") as f:
    f.write(report_content)
f.close()