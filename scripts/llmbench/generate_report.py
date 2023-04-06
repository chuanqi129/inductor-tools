import pandas as pd

commit_list=[]
url_list=[]
result = pd.read_table('result.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
componment = ["benchmark","pytorch","vision","text","audio","data"]
for item in componment:
    sha_short = result.loc[componment.index(item), "commit"][-7:] if item != "benchmark" \
        else result.loc[componment.index(item),"commit"][-8:]
    commit_list.append(sha_short)    
    url_list.append(f"https://github.com/pytorch/{item}/commit/"+sha_short) 
precision = result.loc[7,"commit"]
latency = result.loc[8,"commit"]
transformers = result.loc[6,"commit"]

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
            <th>latency</th> \
        </tr> \
        <tr> \
            <td><p style="text-align:center">{precision}</p></td> \
            <td><p style="text-align:center">32</p></td> \
            <td><p style="text-align:center">True</p></td> \
            <td><p style="text-align:center">True</p></td> \
            <td><p style="text-align:center">{latency}</p></td> \                                   
        </tr> \
    </table> \
    <table border="1"> \
    <p>SW Info:</p> \
        <tr><td>Pytorch:&nbsp;</td><td><a href={url_list[1]}> {commit_list[1]}</a></td></tr> \
        <tr><td>transformers:&nbsp;</td><td>{transformers}</td></tr> \
        <tr><td>TORCH_VISION:&nbsp;</td><td><a href={url_list[2]}> {commit_list[2]} </a></td></tr> \
        <tr><td>TORCH_TEXT:&nbsp;</td><td><a href={url_list[3]}> {commit_list[3]} </a></td></tr> \
        <tr><td>TORCH_AUDIO:&nbsp;</td><td><a href={url_list[4]}> {commit_list[4]} </a></td></tr> \
        <tr><td>TORCH_DATA:&nbsp;</td><td><a href={url_list[5]}> {commit_list[5]} </a></td></tr> \
        <tr><td>TORCH_BENCH:&nbsp;</td><td><a href={url_list[0]}> {commit_list[0]} </a></td></tr> \
    </table> \
    <p>HW info:</p><ol><table> \
        <tbody> \
            <tr><td>Machine name:&nbsp;</td><td>mlp-validate-icx24-ubuntu</td></tr> \
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
    <h4>Thanks.</h4> \
</body> \
</html> \
'''

with open("llm_report.html",mode = "a") as f:
    f.write(report_content)
f.close()
