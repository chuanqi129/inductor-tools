import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Torchinductor LLMbench Report Generate')
parser.add_argument('-l', '--url', type=str, help='jenkins build url')
parser.add_argument('-n', '--node', type=str, default='mlp-validate-icx24-ubuntu',help='benchmark node')
args = parser.parse_args()


def calculate_ratio(values,last_values):
    res = ''
    last_values=last_values.split(',')
    values=values.split(',')
    for i in range (len(values)):
        res += str(round(float(values[i])/float(last_values[i]),2)) + ','
    return res

commit_list=[]
url_list=[]
result = pd.read_table('result.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
componment = ["benchmark","pytorch","vision","text","audio","data"]
for item in componment:
    sha_short = result.loc[componment.index(item), "commit"][-7:] if item != "benchmark" \
        else result.loc[componment.index(item),"commit"][-8:]
    commit_list.append(sha_short) 
    url_list.append("https://github.com/pytorch/"+item+"/commit/"+sha_short)
transformers = result.loc[6,"commit"]
precision = result.loc[7,"commit"]

# latency dict : 8 groups
latency_dict = {}
latency_dict["inductor_gptj_default"] = result.loc[8,'item']
latency_dict["inductor_llama_default"] = result.loc[9,'item']
latency_dict["inductor_gptj_cpp"] = result.loc[10,'item']
latency_dict["inductor_llama_cpp"] = result.loc[11,'item']

latency_dict["eager_gptj_default"] = result.loc[12,'item']
latency_dict["eager_llama_default"] = result.loc[13,'item']
latency_dict["eager_gptj_cpp"] = result.loc[14,'item']
latency_dict["eager_llama_cpp"] = result.loc[15,'item']

# inductor speedup: 4 values
gptj_default_inductor_speedup = result.loc[16,'item'].split(',')[0]
gptj_cpp_inductor_speedup = result.loc[16,'item'].split(',')[1]
llama_default_inductor_speedup = result.loc[16,'item'].split(',')[2]
llama_cpp_inductor_speedup = result.loc[16,'item'].split(',')[3]

# lastsucceful
last_latency_dict = {}
last_latency_dict["inductor_gptj_default"] = 'None'
last_latency_dict["inductor_llama_default"] = 'None'
last_latency_dict["inductor_gptj_cpp"] = 'None'
last_latency_dict["inductor_llama_cpp"] = 'None'

last_latency_dict["eager_gptj_default"] = 'None'
last_latency_dict["eager_llama_default"] = 'None'
last_latency_dict["eager_gptj_cpp"] = 'None'
last_latency_dict["eager_llama_cpp"] = 'None'

last_gptj_default_inductor_speedup = 'None'
last_gptj_cpp_inductor_speedup = 'None'
last_llama_default_inductor_speedup = 'None'
last_llama_cpp_inductor_speedup = 'None'

ratio_inductor_gtpj_default = 'None'
ratio_inductor_gtpj_cpp = 'None'
ratio_inductor_llama_default = 'None'
ratio_inductor_llama_cpp = 'None'
ratio_eager_gtpj_default = 'None'
ratio_eager_gtpj_cpp = 'None'
ratio_eager_llama_default = 'None'
ratio_eager_llama_cpp = 'None'
ratio_speedup_gptj_default = 'None'
ratio_speedup_gptj_cpp = 'None'
ratio_speedup_llama_default = 'None'
ratio_speedup_llama_cpp = 'None'

try:
    last_result = pd.read_table('llm_bench_'+args.node+'/result.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
    last_latency_dict["inductor_gptj_default"] = last_result.loc[8,'item']
    last_latency_dict["inductor_llama_default"] = last_result.loc[9,'item']
    last_latency_dict["inductor_gptj_cpp"] = last_result.loc[10,'item']
    last_latency_dict["inductor_llama_cpp"] = last_result.loc[11,'item']

    last_latency_dict["eager_gptj_default"] = last_result.loc[12,'item']
    last_latency_dict["eager_llama_default"] = last_result.loc[13,'item']
    last_latency_dict["eager_gptj_cpp"] = last_result.loc[14,'item']
    last_latency_dict["eager_llama_cpp"] = last_result.loc[15,'item']

    last_gptj_default_inductor_speedup = last_result.loc[16,'item'].split(',')[0]
    last_gptj_cpp_inductor_speedup = last_result.loc[16,'item'].split(',')[1]
    last_llama_default_inductor_speedup = last_result.loc[16,'item'].split(',')[2]
    last_llama_cpp_inductor_speedup = last_result.loc[16,'item'].split(',')[3]

    # round results ratio calculation
    ratio_speedup_gptj_default = round(float(gptj_default_inductor_speedup)/float(last_gptj_default_inductor_speedup),2)
    ratio_speedup_gptj_cpp = round(float(gptj_cpp_inductor_speedup)/float(last_gptj_cpp_inductor_speedup),2)
    ratio_speedup_llama_default = round(float(llama_default_inductor_speedup)/float(last_llama_default_inductor_speedup),2)
    ratio_speedup_llama_cpp = round(float(llama_cpp_inductor_speedup)/float(last_llama_cpp_inductor_speedup),2)
   
    ratio_inductor_gtpj_default = calculate_ratio(last_latency_dict["inductor_gptj_default"],latency_dict["inductor_gptj_default"])
    ratio_inductor_gtpj_cpp =calculate_ratio(last_latency_dict["inductor_gptj_cpp"],latency_dict["inductor_gptj_cpp"])
    ratio_inductor_llama_default = calculate_ratio(last_latency_dict["inductor_llama_default"],latency_dict["inductor_llama_default"])
    ratio_inductor_llama_cpp = calculate_ratio(last_latency_dict["inductor_llama_cpp"],latency_dict["inductor_llama_cpp"])

    ratio_eager_gtpj_default = calculate_ratio(last_latency_dict["eager_gptj_default"],latency_dict["eager_gptj_default"])
    ratio_eager_gtpj_cpp = calculate_ratio(last_latency_dict["eager_gptj_cpp"],latency_dict["eager_gptj_cpp"])
    ratio_eager_llama_default = calculate_ratio(last_latency_dict["eager_llama_default"],latency_dict["eager_llama_default"])
    ratio_eager_llama_cpp = calculate_ratio(last_latency_dict["eager_llama_cpp"],latency_dict["eager_llama_cpp"])



except:
    pass



report_content=f'''<!DOCTYPE html>
<html>

<head>
    <title>LLM Model Report</title>
</head>

<body>
    <h3> LLM Model(GPTJ & LLAMA) Inductor Benchmark Report </h3>
    <p>Result 1: use_dynamo = True</p>
    <table border="1">
        <tr>
            <th>Model</th>
            <th>precision</th>
            <th>cpp_wrapper</th>
            <th>latency[1]/first_latency/avg_latency/p90_latency/p99_latency</th>
            <th>lastsuccessful</th>
            <th>ratio(last / current)</th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['inductor_gptj_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['inductor_gptj_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_inductor_gtpj_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['inductor_gptj_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['inductor_gptj_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_inductor_gtpj_cpp}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['inductor_llama_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['inductor_llama_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_inductor_llama_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['inductor_llama_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['inductor_llama_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_inductor_llama_cpp}</p>
            </td>
        </tr>
    </table>
    <p>Result 2: use_dynamo = False</p>
    <table border="1">
        <tr>
            <th>Model</th>
            <th>precision</th>
            <th>cpp_wrapper</th>
            <th>latency[1]/first_latency/avg_latency/p90_latency/p99_latency</th>
            <th>lastsuccessful</th>
            <th>ratio(last / current)</th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['eager_gptj_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['eager_gptj_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_eager_gtpj_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['eager_gptj_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['eager_gptj_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_eager_gtpj_cpp}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['eager_llama_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['eager_llama_default']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_eager_llama_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{latency_dict['eager_llama_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{last_latency_dict['eager_llama_cpp']}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_eager_llama_cpp}</p>
            </td>
        </tr>
    </table>
    <p>Result 3: Inductor Speedup</p>
    <table border="1">
        <tr>
            <th>Model</th>
            <th>precision</th>
            <th>cpp_wrapper</th>
            <th>speedup</th>
            <th>lastsuccessful</th>
            <th>ratio(current / last)</th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{gptj_default_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{last_gptj_default_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_speedup_gptj_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">gptj6B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{gptj_cpp_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{last_gptj_cpp_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_speedup_gptj_cpp}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">False</p>
            </td>
            <td>
                <p style="text-align:center">{llama_default_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{last_llama_default_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_speedup_llama_default}</p>
            </td>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">llama7B</p>
            </td>
            <td>
                <p style="text-align:center">{precision}</p>
            </td>
            <td>
                <p style="text-align:center">True</p>
            </td>
            <td>
                <p style="text-align:center">{llama_cpp_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{last_llama_cpp_inductor_speedup}</p>
            </td>
            <td>
                <p style="text-align:center">{ratio_speedup_llama_cpp}</p>
            </td>
        </tr>
    </table>
    <p>SW Info:</p>
    <table border="1">
        <tr>
            <td>Pytorch:&nbsp;</td>
            <td><a href={url_list[1]}> {commit_list[1]}</a></td>
        </tr>
        <tr>
            <td>transformers:&nbsp;</td>
            <td>{transformers}</td>
        </tr>
        <tr>
            <td>TORCH_VISION:&nbsp;</td>
            <td><a href={url_list[2]}> {commit_list[2]} </a></td>
        </tr>
        <tr>
            <td>TORCH_TEXT:&nbsp;</td>
            <td><a href={url_list[3]}> {commit_list[3]} </a></td>
        </tr>
        <tr>
            <td>TORCH_AUDIO:&nbsp;</td>
            <td><a href={url_list[4]}> {commit_list[4]} </a></td>
        </tr>
        <tr>
            <td>TORCH_DATA:&nbsp;</td>
            <td><a href={url_list[5]}> {commit_list[5]} </a></td>
        </tr>
        <tr>
            <td>TORCH_BENCH:&nbsp;</td>
            <td><a href={url_list[0]}> {commit_list[0]} </a></td>
        </tr>
    </table>
    <p>HW info:</p>
    <ol>
        <table>
            <tbody>
                <tr>
                    <td>Machine name:&nbsp;</td>
                    <td>{args.node}</td>
                </tr>
            </tbody>
        </table>
    </ol>
    <p>job info:</p>
    <ol>
        <table>
            <tbody>
                <tr>
                    <td>Build url:&nbsp;</td>
                    <td>{args.url}</td>
                </tr>
            </tbody>
        </table>
    </ol>
    <h4>Thanks.</h4>
</body>

</html>
'''

with open("llm_report.html",mode = "a") as f:
    f.write(report_content)
f.close()