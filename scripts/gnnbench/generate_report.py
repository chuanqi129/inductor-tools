import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='Torchinductor GNNbench Report Generate')
parser.add_argument('-l', '--url', type=str, help='jenkins build url')
args = parser.parse_args()

commit_list = []
url_list = []
result = pd.read_table('result.txt', sep='\:', header=None, names=[
                       'item', 'commit'], engine='python')
componment = ["benchmark", "pytorch", "vision", "text", "audio", "data"]
for item in componment:
    sha_short = result.loc[componment.index(item), "commit"][-7:] if item != "benchmark" \
        else result.loc[componment.index(item), "commit"][-8:]
    commit_list.append(sha_short)
    url_list.append("https://github.com/pytorch/"+item+"/commit/"+sha_short)

# version
pyg_lib = result.loc[6, "commit"]
torch_geometric = result.loc[7, "commit"]
ogb = result.loc[8, "commit"]
torch_scatter = result.loc[9, "commit"]
torch_sparse = result.loc[10, "commit"]

# results
# case 1
GCN_Vanilla_Ftime = result.loc[11, "commit"]
GCN_Vanilla_Btime = result.loc[12, "commit"]
GCN_Vanilla_Ttime = result.loc[13, "commit"]
GCN_Compiled_Ftime = result.loc[14, "commit"]
GCN_Compiled_Btime = result.loc[15, "commit"]
GCN_Compiled_Ttime = result.loc[16, "commit"]

GraphSAGE_Vanilla_Ftime = result.loc[17, "commit"]
GraphSAGE_Vanilla_Btime = result.loc[18, "commit"]
GraphSAGE_Vanilla_Ttime = result.loc[19, "commit"]
GraphSAGE_Compiled_Ftime = result.loc[20, "commit"]
GraphSAGE_Compiled_Btime = result.loc[21, "commit"]
GraphSAGE_Compiled_Ttime = result.loc[22, "commit"]

GIN_Vanilla_Ftime = result.loc[23, "commit"]
GIN_Vanilla_Btime = result.loc[24, "commit"]
GIN_Vanilla_Ttime = result.loc[25, "commit"]
GIN_Compiled_Ftime = result.loc[26, "commit"]
GIN_Compiled_Btime = result.loc[27, "commit"]
GIN_Compiled_Ttime = result.loc[28, "commit"]

EdgeCNN_Vanilla_Ftime = result.loc[29, "commit"]
EdgeCNN_Vanilla_Btime = result.loc[30, "commit"]
EdgeCNN_Vanilla_Ttime = result.loc[31, "commit"]
EdgeCNN_Compiled_Ftime = result.loc[32, "commit"]
EdgeCNN_Compiled_Btime = result.loc[33, "commit"]
EdgeCNN_Compiled_Ttime = result.loc[34, "commit"]

GCN_speedup = result.loc[35, "commit"]
GraphSAGE_speedup = result.loc[36, "commit"]
GIN_speedup = result.loc[37, "commit"]
EdgeCNN_speedup = result.loc[38, "commit"]

last_GCN_speedup=0
last_GraphSAGE_speedup=0
last_GIN_speedup=0
last_EdgeCNN_speedup=0

GCN_speedup_ratio = 0
GraphSAGE_speedup_ratio = 0
GIN_speedup_ratio = 0
EdgeCNN_speedup_ratio = 0

# # case 2 failed w/ inductor now
vanilla_gnn_train_accuracy_use_sage = result.loc[39, "commit"]
vanilla_gnn_valid_accuracy_use_sage = result.loc[40, "commit"]

try:
    compiled_gnn_train_accuracy_use_sage=result.loc[41, "commit"]
    compiled_gnn_valid_accuracy_use_sage=result.loc[42, "commit"]
except:
    compiled_gnn_train_accuracy_use_sage = 0
    compiled_gnn_valid_accuracy_use_sage = 0
    pass

last_vanilla_gnn_train_accuracy_use_sage=0
last_vanilla_gnn_valid_accuracy_use_sage=0
last_compiled_gnn_train_accuracy_use_sage=0
last_compiled_gnn_valid_accuracy_use_sage=0

Ratio_vanilla_gnn_train_accuracy = 0
Ratio_vanilla_gnn_valid_accuracy = 0
Ratio_compiled_gnn_train_accuracy = 0
Ratio_compiled_gnn_valid_accuracy = 0

trainning_vanilla_time = result.loc[43, "commit"].split('s')[0]
trainning_compiled_time = result.loc[44, "commit"].split('s')[0]
inference_vanilla_time = result.loc[45, "commit"].split('s')[0]
inference_compiled_time = result.loc[46, "commit"].split('s')[0]

# err msg
inductor_compile_err = result.loc[47, "commit"]

last_trainning_vanilla_time = 0
last_trainning_compiled_time = 0
last_inference_vanilla_time = 0
last_inference_compiled_time = 0

Ratio_trainning_vanilla_time=0
Ratio_trainning_compiled_time=0
Ratio_inference_vanilla_time=0
Ratio_inference_compiled_time=0



try:
    last_result = pd.read_table('gnn_bench/result.txt', sep='\:',
                                header=None, names=['item', 'commit'], engine='python')

    last_GCN_speedup = last_result.loc[35, "commit"]
    last_GraphSAGE_speedup = last_result.loc[36, "commit"]
    last_GIN_speedup = last_result.loc[37, "commit"]
    last_EdgeCNN_speedup = last_result.loc[38, "commit"]
    last_vanilla_gnn_train_accuracy_use_sage=last_result.loc[39, "commit"]
    last_vanilla_gnn_valid_accuracy_use_sage=last_result.loc[40, "commit"]

    GCN_speedup_ratio = float(GCN_speedup) / float(last_GCN_speedup)
    GraphSAGE_speedup_ratio = float(GraphSAGE_speedup) / float(last_GraphSAGE_speedup)
    GIN_speedup_ratio = float(GIN_speedup) / float(last_GIN_speedup)
    EdgeCNN_speedup_ratio = float(EdgeCNN_speedup) / float(last_EdgeCNN_speedup)      
    Ratio_vanilla_gnn_train_accuracy = float(vanilla_gnn_train_accuracy_use_sage) / float(last_vanilla_gnn_train_accuracy_use_sage)
    Ratio_vanilla_gnn_valid_accuracy = float(vanilla_gnn_valid_accuracy_use_sage) / float(last_vanilla_gnn_valid_accuracy_use_sage)

    last_compiled_gnn_train_accuracy_use_sage=last_result.loc[41, "commit"]
    last_compiled_gnn_valid_accuracy_use_sage=last_result.loc[42, "commit"]
    last_trainning_vanilla_time = last_result.loc[43, "commit"].split('s')[0]
    last_trainning_compiled_time = last_result.loc[44, "commit"].split('s')[0]
    last_inference_vanilla_time = last_result.loc[45, "commit"].split('s')[0]
    last_inference_compiled_time = last_result.loc[46, "commit"].split('s')[0]
    
    Ratio_trainning_vanilla_time = float(last_trainning_vanilla_time) / float(trainning_vanilla_time)
    Ratio_inference_vanilla_time=float(last_inference_vanilla_time) / float(inference_vanilla_time)  

    Ratio_compiled_gnn_train_accuracy = float(compiled_gnn_train_accuracy_use_sage) / float(last_compiled_gnn_train_accuracy_use_sage) 
    Ratio_compiled_gnn_valid_accuracy = float(compiled_gnn_valid_accuracy_use_sage) / float(last_compiled_gnn_valid_accuracy_use_sage)
    Ratio_trainning_compiled_time=float(last_trainning_compiled_time) / float(trainning_compiled_time)
    Ratio_inference_compiled_time=float(last_inference_compiled_time) / float(inference_compiled_time)

except:
    pass

report_content = f'''<!DOCTYPE html>
<html>
<head><title>GNN Models Bench Report</title></head>
<body>
    <h3> GNN Models Inductor Benchmark Report </h3>
    <p>PYG Case :</p>
    <table border="1">
        <tr>
            <th>Model</th>
            <th>Mode</th>
            <th>Fwd</th> 
            <th>Bwd</th> 
            <th>Total</th> 
            <th>Speedup</th> 
            <th>Speedup (lastsuccessful)</th> 
            <th>ratio (current/last)</th>
        </tr> 
        <tr>
            <td><p style="text-align:center">GCN</p></td>
            <td><p style="text-align:center">Vanilla</p></td>
            <td><p style="text-align:center">{GCN_Vanilla_Ftime}s</p></td>
            <td><p style="text-align:center">{GCN_Vanilla_Btime}s</p></td>
            <td><p style="text-align:center">{GCN_Vanilla_Ttime}s</p></td>
            <td rowspan="2"><p style="text-align:center">{GCN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{last_GCN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{GCN_speedup_ratio}</p></td>
        </tr>
        <tr> 
            <td><p style="text-align:center">GCN</p></td>
            <td><p style="text-align:center">Compiled</p></td>
            <td><p style="text-align:center">{GCN_Compiled_Ftime}s</p></td>                          
            <td><p style="text-align:center">{GCN_Compiled_Btime}s</p></td>
            <td><p style="text-align:center">{GCN_Compiled_Ttime}s</p></td>                     
        </tr> 
        <tr> 
            <td><p style="text-align:center">GraphSAGE</p></td>
            <td><p style="text-align:center">Vanilla</p></td>
            <td><p style="text-align:center">{GraphSAGE_Vanilla_Ftime}s</p></td>                         
            <td><p style="text-align:center">{GraphSAGE_Vanilla_Btime}s</p></td>                              
            <td><p style="text-align:center">{GraphSAGE_Vanilla_Ttime}s</p></td>
            <td rowspan="2"><p style="text-align:center">{GraphSAGE_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{last_GraphSAGE_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{GraphSAGE_speedup_ratio}</p></td>                                    
        </tr> 
        <tr> 
            <td><p style="text-align:center">GraphSAGE</p></td> 
            <td><p style="text-align:center">Compiled</p></td> 
            <td><p style="text-align:center">{GraphSAGE_Compiled_Ftime}s</p></td>                             
            <td><p style="text-align:center">{GraphSAGE_Compiled_Btime}s</p></td>                             
            <td><p style="text-align:center">{GraphSAGE_Compiled_Ttime}s</p></td>                             
        </tr> 
        <tr> 
            <td><p style="text-align:center">GIN</p></td> 
            <td><p style="text-align:center">Vanilla</p></td> 
            <td><p style="text-align:center">{GIN_Vanilla_Ftime}s</p></td>                              
            <td><p style="text-align:center">{GIN_Vanilla_Btime}s</p></td>                              
            <td><p style="text-align:center">{GIN_Vanilla_Ttime}s</p></td> 
            <td rowspan="2"><p style="text-align:center">{GIN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{last_GIN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{GIN_speedup_ratio}</p></td>
        </tr> 
        <tr> 
            <td><p style="text-align:center">GIN</p></td> 
            <td><p style="text-align:center">Compiled</p></td> 
            <td><p style="text-align:center">{GIN_Compiled_Ftime}s</p></td>                             
            <td><p style="text-align:center">{GIN_Compiled_Btime}s</p></td>                             
            <td><p style="text-align:center">{GIN_Compiled_Ttime}s</p></td>                             
        </tr> 
        <tr> 
            <td><p style="text-align:center">EdgeCNN</p></td>
            <td><p style="text-align:center">Vanilla</p></td>
            <td><p style="text-align:center">{EdgeCNN_Vanilla_Ftime}s</p></td>                              
            <td><p style="text-align:center">{EdgeCNN_Vanilla_Btime}s</p></td>                              
            <td><p style="text-align:center">{EdgeCNN_Vanilla_Ttime}s</p></td> 
            <td rowspan="2"><p style="text-align:center">{EdgeCNN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{last_EdgeCNN_speedup}</p></td>
            <td rowspan="2"><p style="text-align:center">{EdgeCNN_speedup_ratio}</p></td>
        </tr> 
        <tr>
            <td><p style="text-align:center">EdgeCNN</p></td>
            <td><p style="text-align:center">Compiled</p></td>
            <td><p style="text-align:center">{EdgeCNN_Compiled_Ftime}s</p></td>                            
            <td><p style="text-align:center">{EdgeCNN_Compiled_Btime}s</p></td>                       
            <td><p style="text-align:center">{EdgeCNN_Compiled_Ttime}s</p></td>                            
        </tr>
    </table> 
    <p>OGB Case: Training</p>
    <table border="1">
        <tr>
            <th rowspan="4"><p style="text-align:center">Vanilla</p></th>
            <th>item</th>
            <th>accuracy or time cost</th>
            <th>accuracy or time cost(lastsuccessful)</th>
            <th>ratio</th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">trainning_vanilla_e2etime</p>
            </td>
            <td>
                <p style="text-align:center">{trainning_vanilla_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{last_trainning_vanilla_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{Ratio_trainning_vanilla_time}</p>
            </td>
        </tr>
        <tr>
            <td><p style="text-align:center">train_accuracy_use_sage</p></td>
            <td><p style="text-align:center">{vanilla_gnn_train_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{last_vanilla_gnn_train_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{Ratio_vanilla_gnn_train_accuracy}</p></td>
        </tr>
        <tr>
            <td><p style="text-align:center">valid_accuracy_use_sage</p></td>
            <td><p style="text-align:center">{vanilla_gnn_valid_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{last_vanilla_gnn_valid_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{Ratio_vanilla_gnn_valid_accuracy}</p></td>
        </tr>
        <tr>
            <th rowspan="4"><p style="text-align:center">Compiled</p></th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">trainning_compiled_e2etime</p>
            </td>
            <td>
                <p style="text-align:center">{trainning_compiled_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{last_trainning_compiled_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{Ratio_trainning_compiled_time}</p>
            </td>
        </tr>
        <tr>
            <td><p style="text-align:center">train_accuracy_use_sage</p></td>
            <td><p style="text-align:center">{compiled_gnn_train_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{last_compiled_gnn_train_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{Ratio_compiled_gnn_train_accuracy}</p></td>
        </tr>
        <tr>
            <td><p style="text-align:center">valid_accuracy_use_sage</p></td>
            <td><p style="text-align:center">{compiled_gnn_valid_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{last_compiled_gnn_valid_accuracy_use_sage}%</p></td>
            <td><p style="text-align:center">{Ratio_compiled_gnn_valid_accuracy}</p></td>
        </tr>
    </table>
    <p>OGB Case: Inference</p>
    <table border="1">
        <tr>
            <th rowspan="2">
                <p style="text-align:center">Vanilla</p>
            </th>
            <th>item</th>
            <th>time cost</th>
            <th>time cost(lastsuccessful)</th>
            <th>ratio</th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">inference_vanilla_e2etime</p>
            </td>
            <td>
                <p style="text-align:center">{inference_vanilla_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{last_inference_vanilla_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{Ratio_inference_vanilla_time}</p>
            </td>
        </tr>
        <tr>
            <th rowspan="2">
                <p style="text-align:center">Compiled</p>
            </th>
        </tr>
        <tr>
            <td>
                <p style="text-align:center">inference_compiled_e2etime</p>
            </td>
            <td>
                <p style="text-align:center">{inference_compiled_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{last_inference_compiled_time}s</p>
            </td>
            <td>
                <p style="text-align:center">{Ratio_inference_compiled_time}</p>
            </td>
        </tr>
    </table>
    <p><th>inductor_compile_err: {inductor_compile_err}</th></p>
    <p>SW Info:</p> 
    <table border="1"> 
        <tr><td>Pytorch:&nbsp;</td><td><a href={url_list[1]}> {commit_list[1]}</a></td></tr> 
        <tr><td>TORCH_VISION:&nbsp;</td><td><a href={url_list[2]}> {commit_list[2]} </a></td></tr> 
        <tr><td>TORCH_TEXT:&nbsp;</td><td><a href={url_list[3]}> {commit_list[3]} </a></td></tr> 
        <tr><td>TORCH_AUDIO:&nbsp;</td><td><a href={url_list[4]}> {commit_list[4]} </a></td></tr> 
        <tr><td>TORCH_DATA:&nbsp;</td><td><a href={url_list[5]}> {commit_list[5]} </a></td></tr> 
        <tr><td>TORCH_BENCH:&nbsp;</td><td><a href={url_list[0]}> {commit_list[0]} </a></td></tr> 
        <tr><td>pyg_lib:&nbsp;</td><td>{pyg_lib}</td></tr> 
        <tr><td>torch_geometric:&nbsp;</td><td>{torch_geometric}</td></tr> 
        <tr><td>ogb:&nbsp;</td><td>{ogb}</td></tr> 
        <tr><td>torch_sparse:&nbsp;</td><td>{torch_sparse}</td></tr> 
        <tr><td>torch_scatter:&nbsp;</td><td>{torch_scatter}</td></tr> 
    </table> 
    <p>HW info:</p><ol><table> 
        <tbody> 
            <tr><td>Machine name:&nbsp;</td><td>mlp-validate-icx24-ubuntu</td></tr> 
            <tr><td>Manufacturer:&nbsp;</td><td>Intel Corporation</td></tr> 
            <tr><td>Kernel:</td><td>5.4.0-131-generic</td></tr> 
            <tr><td>Microcode:</td><td>0xd000375</td></tr> 
            <tr><td>Installed Memory:</td><td>503GB</td></tr> 
            <tr><td>OS:</td><td>Ubuntu 18.04.6 LTS</td></tr> 
            <tr><td>CPU Model:</td><td>Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz</td></tr> 
            <tr><td>GCC:</td><td>gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0</td></tr> 
            <tr><td>GLIBC:</td><td>ldd (Ubuntu GLIBC 2.27-3ubuntu1.5) 2.27</td></tr> 
            <tr><td>Binutils:</td><td>GNU ld (GNU Binutils for Ubuntu) 2.30</td></tr> 
            <tr><td>Python:</td><td>Python 3.8.3</td></tr> 
        </tbody></table></ol> 
    <p>job info:</p><ol><table>
        <tbody>
            <tr><td>Build url:&nbsp;</td><td>{args.url}</td></tr>
        </tbody></table></ol>         
    <h4>Thanks.</h4> 
</body> 
</html> 
'''

with open("gnn_report.html", mode="a") as f:
    f.write(report_content)
f.close()
