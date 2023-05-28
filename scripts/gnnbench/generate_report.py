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

# results
GCN_Vanilla_time = result.loc[9, "commit"].split("s")[0]
GCN_Compiled_time = result.loc[10, "commit"].split("s")[0]
GraphSAGE_Vanilla_time = result.loc[11, "commit"].split("s")[0]
GraphSAGE_Compiled_time = result.loc[12, "commit"].split("s")[0]
GIN_Vanilla_time = result.loc[13, "commit"].split("s")[0]
GIN_Compiled_time = result.loc[14, "commit"].split("s")[0]
EdgeCNN_Vanilla_time = result.loc[15, "commit"].split("s")[0]
EdgeCNN_Compiled_time = result.loc[16, "commit"].split("s")[0]
gnn_train_accuracy_use_sage = result.loc[17, "commit"]
gnn_valid_accuracy_use_sage = result.loc[18, "commit"]

# last successful result
last_GCN_Vanilla_time = 0
last_GCN_Compiled_time = 0
last_GraphSAGE_Vanilla_time = 0
last_GraphSAGE_Compiled_time = 0
last_GIN_Vanilla_time = 0
last_GIN_Compiled_time = 0
last_EdgeCNN_Vanilla_time = 0
last_EdgeCNN_Compiled_time = 0
last_gnn_train_accuracy_use_sage = 0
last_gnn_valid_accuracy_use_sage = 0


try:
    last_result = pd.read_table('gnn_bench/result.txt', sep='\:',
                                header=None, names=['item', 'commit'], engine='python')

    last_GCN_Vanilla_time = last_result.loc[9, "commit"].split("s")[0]
    last_GCN_Compiled_time = last_result.loc[10, "commit"].split("s")[0]
    last_GraphSAGE_Vanilla_time = last_result.loc[11, "commit"].split("s")[0]
    last_GraphSAGE_Compiled_time = last_result.loc[12, "commit"].split("s")[0]
    last_GIN_Vanilla_time = last_result.loc[13, "commit"].split("s")[0]
    last_GIN_Compiled_time = last_result.loc[14, "commit"].split("s")[0]
    last_EdgeCNN_Vanilla_time = last_result.loc[15, "commit"].split("s")[0]
    last_EdgeCNN_Compiled_time = last_result.loc[16, "commit"].split("s")[0]
    last_gnn_train_accuracy_use_sage = last_result.loc[17, "commit"]
    last_gnn_valid_accuracy_use_sage = last_result.loc[18, "commit"]

    # ratio
    Ratio_GCN_Vanilla = float(last_GCN_Vanilla_time) / float(GCN_Vanilla_time)
    Ratio_GCN_Compiled = float(last_GCN_Compiled_time) / float(GCN_Compiled_time)

    Ratio_GraphSAGE_Vanilla = float(last_GraphSAGE_Vanilla_time) / float(GraphSAGE_Vanilla_time)
    Ratio_GraphSAGE_Compiled = float(last_GraphSAGE_Compiled_time) / float(GraphSAGE_Compiled_time)

    Ratio_GIN_Vanilla = float(last_GIN_Vanilla_time) / float(GIN_Vanilla_time)
    Ratio_GIN_Compiled = float(last_GIN_Compiled_time) / float(GIN_Compiled_time)

    Ratio_EdgeCNN_Vanilla = float(last_EdgeCNN_Vanilla_time) / float(EdgeCNN_Vanilla_time)
    Ratio_EdgeCNN_Compiled = float(last_EdgeCNN_Compiled_time) / float(EdgeCNN_Compiled_time)

    Ratio_gnn_train_accuracy = float(gnn_train_accuracy_use_sage) / float(last_gnn_train_accuracy_use_sage)
    Ratio_gnn_valid_accuracy = float(gnn_valid_accuracy_use_sage) / float(last_gnn_valid_accuracy_use_sage)

except:
    Ratio_GCN_Vanilla = 0
    Ratio_GCN_Compiled = 0
    Ratio_GraphSAGE_Vanilla = 0
    Ratio_GraphSAGE_Compiled = 0
    Ratio_GIN_Vanilla = 0
    Ratio_GIN_Compiled = 0
    Ratio_EdgeCNN_Vanilla = 0
    Ratio_EdgeCNN_Compiled = 0
    Ratio_gnn_train_accuracy = 0
    Ratio_gnn_valid_accuracy = 0 
    pass

report_content = f'''<!DOCTYPE html>
<html>
<head><title>GNN Models Bench Report</title></head>
<body>
    <h3> GNN Models Inductor Benchmark Report </h3>
    <p>Result:</p>
    <table border="1">
        <tr>
            <th>Model/Case</th>
            <th>note</th>
            <th>result</th> 
            <th>result(lastsuccessful)</th> 
            <th>ratio</th>
        </tr> 
        <tr> 
            <td><p style="text-align:center">GCN</p></td> 
            <td><p style="text-align:center">Vanilla_time</p></td> 
            <td><p style="text-align:center">{GCN_Vanilla_time}s</p></td>                                  
            <td><p style="text-align:center">{last_GCN_Vanilla_time}s</p></td>                                 
            <td><p style="text-align:center">{Ratio_GCN_Vanilla}s</p></td>                                 
        </tr> 
        <tr> 
            <td><p style="text-align:center">GCN</p></td> 
            <td><p style="text-align:center">Compiled_time</p></td> 
            <td><p style="text-align:center">{GCN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{last_GCN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{Ratio_GCN_Compiled}s</p></td>                               
        </tr> 

        <tr> 
            <td><p style="text-align:center">GraphSAGE</p></td> 
            <td><p style="text-align:center">Vanilla_time</p></td> 
            <td><p style="text-align:center">{GraphSAGE_Vanilla_time}s</p></td>                                  
            <td><p style="text-align:center">{last_GraphSAGE_Vanilla_time}s</p></td>                                 
            <td><p style="text-align:center">{Ratio_GraphSAGE_Vanilla}s</p></td>                                 
        </tr> 
        <tr> 
            <td><p style="text-align:center">GraphSAGE</p></td> 
            <td><p style="text-align:center">Compiled_time</p></td> 
            <td><p style="text-align:center">{GraphSAGE_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{last_GraphSAGE_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{Ratio_GraphSAGE_Compiled}s</p></td>                               
        </tr> 

        <tr> 
            <td><p style="text-align:center">GIN</p></td> 
            <td><p style="text-align:center">Vanilla_time</p></td> 
            <td><p style="text-align:center">{GIN_Vanilla_time}s</p></td>                                  
            <td><p style="text-align:center">{last_GIN_Vanilla_time}s</p></td>                                 
            <td><p style="text-align:center">{Ratio_GIN_Vanilla}s</p></td>                                 
        </tr> 
        <tr> 
            <td><p style="text-align:center">GIN</p></td> 
            <td><p style="text-align:center">Compiled_time</p></td> 
            <td><p style="text-align:center">{GIN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{last_GIN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{Ratio_GIN_Compiled}s</p></td>                               
        </tr> 

        <tr> 
            <td><p style="text-align:center">EdgeCNN</p></td> 
            <td><p style="text-align:center">Vanilla_time</p></td> 
            <td><p style="text-align:center">{EdgeCNN_Vanilla_time}s</p></td>                                  
            <td><p style="text-align:center">{last_EdgeCNN_Vanilla_time}s</p></td>                                 
            <td><p style="text-align:center">{Ratio_EdgeCNN_Vanilla}s</p></td>                                 
        </tr> 
        <tr>
            <td><p style="text-align:center">EdgeCNN</p></td> 
            <td><p style="text-align:center">Compiled_time</p></td> 
            <td><p style="text-align:center">{EdgeCNN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{last_EdgeCNN_Compiled_time}s</p></td>                                    
            <td><p style="text-align:center">{Ratio_EdgeCNN_Compiled}</p></td>                               
        </tr>
        <tr> 
            <td><p style="text-align:center">ogb example</p></td> 
            <td><p style="text-align:center">train_accuracy_use_sage</p></td> 
            <td><p style="text-align:center">{gnn_train_accuracy_use_sage}%</p></td>                                  
            <td><p style="text-align:center">{last_gnn_train_accuracy_use_sage}%</p></td>                                 
            <td><p style="text-align:center">{Ratio_gnn_train_accuracy}</p></td>                                 
        </tr> 
        <tr> 
            <td><p style="text-align:center">ogb example</p></td> 
            <td><p style="text-align:center">valid_accuracy_use_sage</p></td> 
            <td><p style="text-align:center">{gnn_valid_accuracy_use_sage}%</p></td>                                    
            <td><p style="text-align:center">{last_gnn_valid_accuracy_use_sage}%</p></td>                                    
            <td><p style="text-align:center">{Ratio_gnn_valid_accuracy}</p></td>                                   
        </tr>
    </table> 
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
