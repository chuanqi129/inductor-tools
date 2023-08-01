"""
Generate report or data compare report from specified inductor logs.
Usage:
  python report.py -r WW48.2 -t WW48.4 -m all --html_off --md_off --precision bfloat16
  python report.py -r WW48.2 -t WW48.4 -m all --gh_token github_pat_xxxxx --dashboard dynamic
Dependencies:
    styleframe
    PyGithub
"""

import argparse
from datetime import datetime,timedelta
from styleframe import StyleFrame, Styler, utils
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description="Generate report from two specified inductor logs")
parser.add_argument('-t','--target',type=str,help='target log file')
parser.add_argument('-r','--reference',type=str,help='reference log file')
parser.add_argument('-l', '--url', type=str, help='jenkins job build url')
parser.add_argument('-m','--mode',type=str,help='multiple or single mode')
parser.add_argument('-p','--precision',type=str,default='float32',help='precision')
parser.add_argument('--md_off', action='store_true', help='turn off markdown files generate')
parser.add_argument('--html_off', action='store_true', help='turn off html file generate')
parser.add_argument('--gh_token', type=str,help='github token for issue comment creation')
parser.add_argument('--dashboard', type=str,default='default',help='determine title in dashboard report')
args=parser.parse_args()

# known failure @20230423
known_failures ={
    "hf_T5_base":"TIMEOUT",
    "pnasnet5large":"ERROR", # dynamic shapes
    "gat":"ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'",
    "gcn":"ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'",
    "sage":"ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'",
    "torchrec_dlrm":"AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'jagged_2d_to_dense'",
}

# SW info
torch_commit=''
torchbench_commit=''
torchaudio_commit=''
torchtext_commit=''
torchvision_commit=''
torchdata_commit=''
dynamo_benchmarks_commit=''

torch_main_commit=''
torchaudio_main_commit=''
torchtext_main_commit=''
torchvision_main_commit=''
torchdata_main_commit=''

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                return os.path.join(root, d)
        for f in files:
            if thread in (os.path.join(root, f)):
                return os.path.join(root, f)                 

if args.reference is not None:
    reference_mt=getfolder(args.reference,'multi_threads_cf_logs')
    reference_st=getfolder(args.reference,'single_thread_cf_logs')
target_mt=getfolder(args.target,'multi_threads_cf_logs')
target_st=getfolder(args.target,'single_thread_cf_logs')

target_style = Styler(bg_color='#DCE6F1', font_color=utils.colors.black)
red_style = Styler(bg_color='#FF0000', font_color=utils.colors.black)
regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)

passed_style = Styler(bg_color='#D8E4BC', font_color=utils.colors.black)
failed_style = Styler(bg_color='#FFC7CE', font_color=utils.colors.black)

def update_summary(excel,reference,target):
    data = {
        'Test Secnario':['Single Socket Multi-Threads', ' ', ' ', ' ','Single Core Single-Thread',' ',' ',' '], 
        'Comp Item':['Pass Rate', ' ', 'Geomean Speedup', ' ','Pass Rate',' ','Geomean Speedup',' '],
        'Date':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor','inductor','inductor','inductor'],
        'torchbench':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'huggingface':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'timm_models ':[' ', ' ', ' ', ' ',' ',' ',' ',' ']
    }
    data_target = {
        'Test Secnario':['Single Socket Multi-Threads', ' ','Single Core Single-Thread',' '], 
        'Comp Item':['Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup'],
        'Date':[' ', ' ', ' ', ' '],
        'Compiler':['inductor', 'inductor', 'inductor', 'inductor'],
        'torchbench':[' ', ' ', ' ', ' '],
        'huggingface':[' ', ' ', ' ', ' '],
        'timm_models ':[' ', ' ', ' ', ' ']
    }    
    if reference is not None:
        summary=pd.DataFrame(data)
        if args.mode == "multiple" or args.mode == 'all':
            reference_mt_pr_data=pd.read_csv(reference_mt+'/passrate.csv',index_col=0)
            reference_mt_gm_data=pd.read_csv(reference_mt+'/geomean.csv',index_col=0)
            summary.iloc[0:1,4:7]=reference_mt_pr_data.iloc[0:2,1:7]
            summary.iloc[2:3,4:7]=reference_mt_gm_data.iloc[0:2,1:7] 
            summary.iloc[0:1,2]=reference
            summary.iloc[2:3,2]=reference 
            target_mt_pr_data=pd.read_csv(target_mt+'/passrate.csv',index_col=0)
            target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0) 
            summary.iloc[1:2,4:7]=target_mt_pr_data.iloc[0:2,1:7]
            summary.iloc[3:4,4:7]=target_mt_gm_data.iloc[0:2,1:7]
            summary.iloc[1:2,2]=target
            summary.iloc[3:4,2]=target            
        if args.mode == "single" or args.mode == 'all':
            reference_st_pr_data=pd.read_csv(reference_st+'/passrate.csv',index_col=0)
            reference_st_gm_data=pd.read_csv(reference_st+'/geomean.csv',index_col=0)
            summary.iloc[4:5,4:7]=reference_st_pr_data.iloc[0:2,1:7]
            summary.iloc[6:7,4:7]=reference_st_gm_data.iloc[0:2,1:7]
            summary.iloc[4:5,2]=reference
            summary.iloc[6:7,2]=reference
            target_st_pr_data=pd.read_csv(target_st+'/passrate.csv',index_col=0)
            target_st_gm_data=pd.read_csv(target_st+'/geomean.csv',index_col=0)
            summary.iloc[5:6,4:7]=target_st_pr_data.iloc[0:2,1:7]
            summary.iloc[7:8,4:7]=target_st_gm_data.iloc[0:2,1:7]
            summary.iloc[5:6,2]=target
            summary.iloc[7:8,2]=target
        sf = StyleFrame(summary)
        sf.apply_style_by_indexes(sf.index[[1,3,5,7]], styler_obj=target_style)         
    else:
        summary=pd.DataFrame(data_target)
        if args.mode == "multiple" or args.mode == 'all':
            target_mt_pr_data=pd.read_csv(target_mt+'/passrate.csv',index_col=0)
            target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0)
            summary.iloc[0:1,4:7]=target_mt_pr_data.iloc[0:2,1:7]
            summary.iloc[1:2,4:7]=target_mt_gm_data.iloc[0:2,1:7] 
            summary.iloc[0:1,2]=target
            summary.iloc[1:2,2]=target
        if args.mode == "single" or args.mode == 'all':
            target_st_pr_data=pd.read_csv(target_st+'/passrate.csv',index_col=0)
            target_st_gm_data=pd.read_csv(target_st+'/geomean.csv',index_col=0)
            summary.iloc[2:3,4:7]=target_st_pr_data.iloc[0:2,1:7]
            summary.iloc[3:4,4:7]=target_st_gm_data.iloc[0:2,1:7]
            summary.iloc[2:3,2]=target
            summary.iloc[3:4,2]=target
        sf = StyleFrame(summary)
    for i in range(1,8):
        sf.set_column_width(i, 18)
    sf.to_excel(sheet_name='Summary',excel_writer=excel)

def get_main_commit(item,nightly_commit):
    input_url="https://github.com/pytorch/"+item+"/commit/"+nightly_commit
    page=requests.get(input_url)
    soup = BeautifulSoup(page.text,features="html.parser")
    item = str(soup.find('title'))
    output=(item.split('(')[1].split(')')[0])[:7]
    return output

def update_swinfo(excel):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata','dynamo_benchmarks'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' ',' '],'Main commit':[' ', ' ', ' ', ' ',' ',' ','/']}
    swinfo=pd.DataFrame(data)
    try:
        version = pd.read_table(args.target+'/inductor_log/version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
        global torch_commit,torchbench_commit,torchaudio_commit,torchtext_commit,torchvision_commit,torchdata_commit,dynamo_benchmarks_commit
        global torch_main_commit,torchaudio_main_commit,torchtext_main_commit,torchvision_main_commit,torchdata_main_commit

        torch_commit=version.loc[ 1, "commit"][-7:]
        torchbench_commit=version.loc[ 0, "commit"][-8:]
        torchaudio_commit=version.loc[ 4, "commit"][-7:]
        torchtext_commit=version.loc[ 3, "commit"][-7:]
        torchvision_commit=version.loc[ 2, "commit"][-7:]
        torchdata_commit=version.loc[ 5, "commit"][-7:]
        dynamo_benchmarks_commit=version.loc[ 6, "commit"][-7:]

        swinfo.loc[0,"Nightly commit"]=torch_commit
        swinfo.loc[1,"Main commit"]=torchbench_commit
        swinfo.loc[2,"Nightly commit"]=torchaudio_commit
        swinfo.loc[3,"Nightly commit"]=torchtext_commit
        swinfo.loc[4,"Nightly commit"]=torchvision_commit
        swinfo.loc[5,"Nightly commit"]=torchdata_commit
        swinfo.loc[6,"Nightly commit"]=dynamo_benchmarks_commit

        torch_main_commit=get_main_commit("pytorch",torch_commit)
        torchaudio_main_commit=get_main_commit("audio",torchaudio_commit)
        torchtext_main_commit=get_main_commit("text",torchtext_commit)
        torchvision_main_commit=get_main_commit("vision",torchvision_commit)
        torchdata_main_commit=get_main_commit("data",torchdata_commit) 

        swinfo.loc[0,"Main commit"]=torch_main_commit
        swinfo.loc[2,"Main commit"]=torchaudio_main_commit
        swinfo.loc[3,"Main commit"]=torchtext_main_commit
        swinfo.loc[4,"Main commit"]=torchvision_main_commit
        swinfo.loc[5,"Main commit"]=torchdata_main_commit
    except :
        print("version.txt not found")
        pass

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)   

# only accuracy failures
def parse_acc_failure(file,failed_model):
    result = []
    found = False
    skip = False
    if failed_model in known_failures.keys():
        result.append(failed_model+", "+known_failures[failed_model])
    else:
        with open(file, 'r') as reader:
            contents = reader.readlines()
            for line in contents:
                # skip performance part
                if "/opt/conda/bin/python -u benchmarks/dynamo/torchbench.py --accuracy" not in line and skip ==False:
                    continue
                skip = True
                if found ==  False and "cpu  eval  " in line:
                    model = line.split("cpu  eval  ")[1].split(' ')[0].strip()
                    if model != failed_model:
                        continue
                    found =  True
                if found ==  True and ("Error: " in line or "[ERROR]" in line or "TIMEOUT" in line or "FAIL" in line):
                    line=line.replace(',',' ',20)
                    result.append(model+", "+ line)
                    break
    return result

# other failures
def parse_failure(file,failed_model):
    result = []
    found = False
    if failed_model in known_failures.keys():
        result.append(failed_model+", "+known_failures[failed_model])
    else:
        with open(file, 'r') as reader:
            contents = reader.readlines()
            for line in contents:
                if found ==  False and "cpu  eval  " in line:
                    model = line.split("cpu  eval  ")[1].split(' ')[0].strip()
                    if model != failed_model:
                        continue
                    found =  True
                elif found ==  True and line.find("Error: ")!=-1 or line.find("TIMEOUT")!=-1:
                    if line.find("Error: ")!=-1:
                        line=line.replace(',',' ',20)
                    result.append(model+", "+ line)
                    break
    return result

def str_to_dict(contents):
    res_dict = {}
    for line in contents:
        model = line.split(", ")[0].strip()
        reason = line.strip()
        res_dict[model] = reason
    return res_dict

def failures_reason_parse(model,acc_tag,mode):
    raw_log_pre="multi_threads_model_bench_log" if "multi" in mode else "single_thread_model_bench_log"
    raw_log=getfolder(args.target,raw_log_pre)
    content=parse_acc_failure(raw_log,model) if acc_tag =="X" else parse_failure(raw_log,model)
    ct_dict=str_to_dict(content)
    try:
        line = ct_dict[model]
    except KeyError:
        line=''
        pass
    return line

def update_failures(excel,target_thread):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=target_thread+'/inductor_'+suite+'_'+args.precision+'_inference_cpu_performance.csv'
        acc_path=target_thread+'/inductor_'+suite+'_'+args.precision+'_inference_cpu_accuracy.csv'

        perf_data=pd.read_csv(perf_path)
        acc_data=pd.read_csv(acc_path)
        
        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='infra_error') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
        acc_data.insert(loc=2, column='acc_suite', value=suite)
        tmp.append(acc_data)

        perf_data=perf_data.loc[(perf_data['batch_size'] ==0) | (perf_data['speedup'] ==0) | (perf_data['speedup'] =='infra_error'),:]
        perf_data.insert(loc=3, column='pef_suite', value=suite)
        tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['acc_suite','pef_suite','name','accuracy','speedup']]
    failures['pef_suite'].fillna(0,inplace=True)
    failures['acc_suite'].fillna(0,inplace=True)
    failures['suite'] = failures.apply(lambda x: x['pef_suite'] if x['acc_suite']==0 else x['acc_suite'], axis=1) 
    failures=failures.rename(columns={'suite':'suite','name':'name','accuracy':'accuracy','speedup':'perf'}) 

    # 1 -> failed
    failures['accuracy'].replace(['infra_error','timeout','fail_to_run','fail_accuracy','0.0000'],[1,1,1,1,1],inplace=True)
    failures['perf'].replace([0,'infra_error','timeout'],[1,1,1],inplace=True)
    failures['suite'].replace(["torchbench","huggingface","timm_models"],[3,4,5],inplace=True)   
    failures=failures.groupby(by=['name']).sum().reset_index()
    failures['suite'].replace([3,4,5,6,8,10],["torchbench","huggingface","timm_models","torchbench","huggingface","timm_models"],inplace=True)
    failures['perf'].replace([0,1],["√","X"],inplace=True)
    failures['accuracy'].replace([0,1],["√","X"],inplace=True)

    # fill failure reasons
    failures['name']=failures['name'].drop_duplicates()
    failures_dict =  {key:values for key, values in zip(failures['name'], failures['accuracy'])}
    reason_content=[]
    for model in failures_dict.keys():
        reason_content.append(failures_reason_parse(model,failures_dict[model],target_thread))
    failures['reason(reference only)']=reason_content

    sf = StyleFrame({'suite': list(failures['suite']),
                 'name': list(failures['name']),
                 'accuracy': list(failures['accuracy']),
                 'perf': list(failures['perf']),
                 'reason(reference only)':(list(failures['reason(reference only)']))})
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['accuracy'] == "X"],
                            cols_to_style='accuracy',
                            styler_obj=failed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['perf'] == "X"],
                            cols_to_style='perf',
                            styler_obj=failed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['accuracy'] == "√"],
                            cols_to_style='accuracy',
                            styler_obj=passed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['perf'] == "√"],
                            cols_to_style='perf',
                            styler_obj=passed_style,
                            overwrite_default_style=False)    
    sf.set_column_width(1, 20)
    sf.set_column_width(2, 30)
    sf.set_column_width(3, 15)
    sf.set_column_width(4, 15)
    sf.set_column_width(5, 100)              

    sf.to_excel(sheet_name='Failures in '+target_thread.split('_cf')[0].split('inductor_log/')[1].strip(),excel_writer=excel,index=False)

def process_suite(suite,thread):
    target_file_path=getfolder(args.target,thread)+'/inductor_'+suite+'_'+args.precision+'_inference_cpu_performance.csv'
    target_ori_data=pd.read_csv(target_file_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency']]
    target_data=target_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)

    if args.reference is not None:
        reference_file_path=getfolder(args.reference,thread)+'/inductor_'+suite+'_'+args.precision+'_inference_cpu_performance.csv'
        reference_ori_data=pd.read_csv(reference_file_path,index_col=0)
        reference_data=reference_ori_data[['name','batch_size','speedup','abs_latency']]
        reference_data=reference_data.copy()
        reference_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)    
        data=pd.merge(target_data,reference_data,on=['name'],how= 'outer')
        return data
    else:
        return target_data

def process_thread(thread):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        data=process_suite(suite, thread)
        tmp.append(data)
    return pd.concat(tmp)
 
def process(input):
    if args.reference is not None:
        data_new=input[['name','batch_size_x','speedup_x','abs_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new'})
        data_new['inductor_new']=data_new['inductor_new'].astype(float).div(1000)
        data_new['speed_up_new']=data_new['speed_up_new'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager_new'] = data_new['speed_up_new'] * data_new['inductor_new']        
        data_old=input[['batch_size_y','speedup_y','abs_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old'})    
        data_old['inductor_old']=data_old['inductor_old'].astype(float).div(1000)
        data_old['speed_up_old']=data_old['speed_up_old'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_old['eager_old'] = data_old['speed_up_old'] * data_old['inductor_old']
        input['speedup_x']=input['speedup_x'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        input['speedup_y']=input['speedup_y'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_ratio= pd.DataFrame(round(input['speedup_x'] / input['speedup_y'],2),columns=['Ratio Speedup(New/old)'])
        data_ratio['Eager Ratio(old/new)'] = pd.DataFrame(round(data_old['eager_old'] / data_new['eager_new'],2))
        data_ratio['Inductor Ratio(old/new)'] = pd.DataFrame(round(data_old['inductor_old'] / data_new['inductor_new'],2))

        data = StyleFrame({'name': list(data_new['name']),
                    'batch_size_new': list(data_new['batch_size_new']),
                    'speed_up_new': list(data_new['speed_up_new']),
                    'inductor_new': list(data_new['inductor_new']),
                    'eager_new': list(data_new['eager_new']),
                    'batch_size_old': list(data_old['batch_size_old']),
                    'speed_up_old': list(data_old['speed_up_old']),
                    'inductor_old': list(data_old['inductor_old']),
                    'eager_old': list(data_old['eager_old']),
                    'Ratio Speedup(New/old)': list(data_ratio['Ratio Speedup(New/old)']),
                    'Eager Ratio(old/new)': list(data_ratio['Eager Ratio(old/new)']),
                    'Inductor Ratio(old/new)': list(data_ratio['Inductor Ratio(old/new)'])})
        data.set_column_width(1, 10)
        data.set_column_width(2, 18) 
        data.set_column_width(3, 18) 
        data.set_column_width(4, 18)
        data.set_column_width(5, 15)
        data.set_column_width(6, 18)
        data.set_column_width(7, 18) 
        data.set_column_width(8, 18) 
        data.set_column_width(9, 15)
        data.set_column_width(10, 28)
        data.set_column_width(11, 28) 
        data.set_column_width(12, 28) 
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size_new'] == 0], styler_obj=red_style)
        data.apply_style_by_indexes(indexes_to_style=data[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < 0.9)],styler_obj=regression_style)
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] > 1.1],styler_obj=improve_style)
        data.set_row_height(rows=data.row_indexes, height=15)    
    else:
        data_new=input[['name','batch_size','speedup','abs_latency']].rename(columns={'name':'name','batch_size':'batch_size','speedup':'speedup',"abs_latency":'inductor'})
        data_new['inductor']=data_new['inductor'].astype(float).div(1000)
        data_new['speedup']=data_new['speedup'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager'] = data_new['speedup'] * data_new['inductor']        
        data = StyleFrame({'name': list(data_new['name']),
                    'batch_size': list(data_new['batch_size']),
                    'speedup': list(data_new['speedup']),
                    'inductor': list(data_new['inductor']),
                    'eager': list(data_new['eager'])})
        data.set_column_width(1, 10)
        data.set_column_width(2, 18) 
        data.set_column_width(3, 18) 
        data.set_column_width(4, 18)
        data.set_column_width(5, 15)
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size'] == 0], styler_obj=red_style)
        data.set_row_height(rows=data.row_indexes, height=15)        
    return data

def update_details(writer):
    h = {"A": 'Model suite',"B": '', "C": args.target, "D": '', "E": '',"F": '', "G": args.reference, "H": '', "I": '',"J": '',"K": 'Result Comp',"L": '',"M": ''}
    if args.reference is None:
        h = {"A": 'Model suite',"B": '', "C": args.target, "D": '', "E": '',"F": ''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    if args.mode == "multiple" or args.mode == 'all':
        # mt
        head.to_excel(excel_writer=writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
        mt=process_thread('multi_threads_cf_logs')
        mt_data=process(mt)

        global torchbench_index
        torchbench_index=mt_data.loc[mt_data['name']=='alexnet'].index[0]
        global hf_index
        hf_index=mt_data.loc[mt_data['name']=='AlbertForMaskedLM'].index[0]
        global timm_index
        timm_index=mt_data.loc[mt_data['name']=='adv_inception_v3'].index[0]
        global end_index
        end_index=len(list(mt_data['name']))-1

        suite_list=[''] * len(list(mt_data['name']))

        suite_list.insert(int(torchbench_index), 'Torchbench')
        suite_list.insert(int(hf_index), 'HF')
        suite_list.insert(int(timm_index), 'Timm')

        s= pd.Series(suite_list)
        s.to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=0)
        mt_data.to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=1)    
    if args.mode == "single" or args.mode == 'all':
        # st
        head.to_excel(excel_writer=writer, sheet_name='Single-Core Single-thread', index=False,startrow=0,header=False) 
        st=process_thread('single_thread_cf_logs')
        st_data=process(st)

        torchbench_index=st_data.loc[st_data['name']=='alexnet'].index[0]
        hf_index=st_data.loc[st_data['name']=='AlbertForMaskedLM'].index[0]
        timm_index=st_data.loc[st_data['name']=='adv_inception_v3'].index[0]
        end_index=len(list(st_data['name']))-1
        suite_list=[''] * len(list(st_data['name']))

        suite_list.insert(int(torchbench_index), 'Torchbench')
        suite_list.insert(int(hf_index), 'HF')
        suite_list.insert(int(timm_index), 'Timm')

        s= pd.Series(suite_list)
        s.to_excel(sheet_name='Single-Core Single-thread',excel_writer=writer,index=False,startrow=1,startcol=0)
        st_data.to_excel(sheet_name='Single-Core Single-thread',excel_writer=writer,index=False,startrow=1,startcol=1)   

def update_issue_commits():
    from github import Github
    # generate md files
    mt_addtional=f'''
SW information:

SW	| Nightly commit	| Main commit
-- | -- | --
Pytorch|[{torch_commit}](https://github.com/pytorch/pytorch/commit/{torch_commit})|[{torch_main_commit}](https://github.com/pytorch/pytorch/commit/{torch_main_commit})
Torchbench|/|[{torchbench_commit}](https://github.com/pytorch/benchmark/commit/{torchbench_commit})
torchaudio|[{torchaudio_commit}](https://github.com/pytorch/audio/commit/{torchaudio_commit})|[{torchaudio_main_commit}](https://github.com/pytorch/audio/commit/{torchaudio_main_commit})
torchtext|[{torchtext_commit}](https://github.com/pytorch/text/commit/{torchtext_commit})| [{torchtext_main_commit}](https://github.com/pytorch/text/commit/{torchtext_main_commit})
torchvision|[{torchvision_commit}](https://github.com/pytorch/vision/commit/{torchvision_commit})|[{torchvision_main_commit}](https://github.com/pytorch/vision/commit/{torchvision_main_commit})
torchdata|[{torchdata_commit}](https://github.com/pytorch/data/commit/{torchdata_commit})|[{torchdata_main_commit}](https://github.com/pytorch/data/commit/{torchdata_main_commit})
dynamo_benchmarks|[{dynamo_benchmarks_commit}](https://github.com/pytorch/pytorch/commit/{dynamo_benchmarks_commit})|/


HW information

|  Item  | Value  |
|  ----  | ----  |
| Manufacturer  | Amazon EC2 |
| Product Name  | c6i.16xlarge |
| CPU Model  | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz |
| Installed Memory  | 128GB (1x128GB DDR4 3200 MT/s [Unknown]) |
| OS  | Ubuntu 22.04.2 LTS |
| Kernel  | 5.19.0-1022-aws |
| Microcode  | 0xd000389 |
| GCC  | gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0 |
| GLIBC  | ldd (Ubuntu GLIBC 2.35-0ubuntu3.1) 2.35 |
| Binutils  | GNU ld (GNU Binutils for Ubuntu) 2.38 |
| Python  | Python 3.10.6 |
| OpenSSL  | OpenSSL 3.0.2 15 Mar 2022 (Library: OpenSSL 3.0.2 15 Mar 2022) |
'''+'''
Test command

```bash
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export TORCHINDUCTOR_FREEZING=1
CORES=$(lscpu | grep Core | awk '{print $4}')
export OMP_NUM_THREADS=$CORES
python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --devices=cpu --dtypes=float32 --inference --compilers=inductor --extra-args="--timeout 9000" 

```
'''
    st_addtional=f'''
SW information:

SW	| Nightly commit	| Main commit
-- | -- | --
Pytorch|[{torch_commit}](https://github.com/pytorch/pytorch/commit/{torch_commit})|[{torch_main_commit}](https://github.com/pytorch/pytorch/commit/{torch_main_commit})
Torchbench|/|[{torchbench_commit}](https://github.com/pytorch/benchmark/commit/{torchbench_commit})
torchaudio|[{torchaudio_commit}](https://github.com/pytorch/audio/commit/{torchaudio_commit})|[{torchaudio_main_commit}](https://github.com/pytorch/audio/commit/{torchaudio_main_commit})
torchtext|[{torchtext_commit}](https://github.com/pytorch/text/commit/{torchtext_commit})| [{torchtext_main_commit}](https://github.com/pytorch/text/commit/{torchtext_main_commit})
torchvision|[{torchvision_commit}](https://github.com/pytorch/vision/commit/{torchvision_commit})|[{torchvision_main_commit}](https://github.com/pytorch/vision/commit/{torchvision_main_commit})
torchdata|[{torchdata_commit}](https://github.com/pytorch/data/commit/{torchdata_commit})|[{torchdata_main_commit}](https://github.com/pytorch/data/commit/{torchdata_main_commit})
dynamo_benchmarks|[{dynamo_benchmarks_commit}](https://github.com/pytorch/pytorch/commit/{dynamo_benchmarks_commit})|/

HW information

|  Item  | Value  |
|  ----  | ----  |
| Manufacturer  | Amazon EC2 |
| Product Name  | c6i.16xlarge |
| CPU Model  | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz |
| Installed Memory  | 128GB (1x128GB DDR4 3200 MT/s [Unknown]) |
| OS  | Ubuntu 22.04.2 LTS |
| Kernel  | 5.19.0-1022-aws |
| Microcode  | 0xd000389 |
| GCC  | gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0 |
| GLIBC  | ldd (Ubuntu GLIBC 2.35-0ubuntu3.1) 2.35 |
| Binutils  | GNU ld (GNU Binutils for Ubuntu) 2.38 |
| Python  | Python 3.10.6 |
| OpenSSL  | OpenSSL 3.0.2 15 Mar 2022 (Library: OpenSSL 3.0.2 15 Mar 2022) |
'''+'''
Test command

```bash
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export TORCHINDUCTOR_FREEZING=1
export OMP_NUM_THREADS=1

python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --devices=cpu --dtypes=float32 --inference --compilers=inductor --batch_size=1 --threads 1 --extra-args="--timeout 9000"

```
'''
    if not args.md_off:
        # mt
        mt_result=open(args.target+'/inductor_log/mt_'+args.target+'.md','a+')
        mt_folder = getfolder(args.target,'multi_threads_cf_logs')
        mt_title=f'# [{args.dashboard}] Performance Dashboard for {args.precision} precision -- Single-Socket Multi-threads ('+str((datetime.now() - timedelta(days=2)).date())+' nightly release) ##'
        mt_result.writelines(mt_title)

        mt_summary=mt_folder+'/gh_executive_summary.txt'
        with open(mt_summary,'r') as summary_file:
            lines=list(summary_file.readlines())
            lines.insert(11, mt_addtional)
            s=' '.join(lines)
            mt_result.writelines(s)

        mt_inference=mt_folder+'/gh_inference.txt'
        with open(mt_inference,'r') as inference_file:
            mt_result.writelines(inference_file.readlines())
        # st
        st_result=open(args.target+'/inductor_log/st_'+args.target+'.md','a+')
        st_folder = getfolder(args.target,'single_thread_cf_logs')
        st_title=f'# [{args.dashboard}] Performance Dashboard for {args.precision} precision -- Single-core Single-thread ('+str((datetime.now() - timedelta(days=2)).date())+' nightly release) ##'
        st_result.writelines(st_title)

        st_summary=st_folder+'/gh_executive_summary.txt'
        with open(st_summary,'r') as summary_file:
            lines=list(summary_file.readlines())
            lines.insert(11, st_addtional)
            s=' '.join(lines)
            st_result.writelines(s)

        st_inference=st_folder+'/gh_inference.txt'
        with open(st_inference,'r') as inference_file:
            st_result.writelines(inference_file.readlines())
        # create comment in github issue
        ESI_SYD_TK = args.gh_token
        g = Github(ESI_SYD_TK)
        g = Github(base_url="https://api.github.com", login_or_token=ESI_SYD_TK)
        repo = g.get_repo("pytorch/pytorch")
        issue = repo.get_issue(number=93531)
        print(issue)
        try:
            mt_result.seek(0)
            st_result.seek(0)
            issue.create_comment(mt_result.read())
            issue.create_comment(st_result.read())
        except:
            print("issue commit create failed")
            pass

def html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<title> Inductor Regular Model Bench Report </title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
<link rel="stylesheet" type="text/css" href="css/font-awesome.min.css">
<link rel="stylesheet" type="text/css" href="css/animate.css">
<link rel="stylesheet" type="text/css" href="css/select2.min.css">
<link rel="stylesheet" type="text/css" href="css/perfect-scrollbar.css">
<link rel="stylesheet" type="text/css" href="css/util.css">
<link rel="stylesheet" type="text/css" href="css/main.css">
<meta name="robots" content="noindex, follow">
</head>
<body>
  <div class="limiter">
  <div class="container-table100">
  <div class="wrap-table100">
  <div class="table100">
  <p><h3>Inductor Regular Model Bench Report </p></h3> '''

def html_tail():
    return f'''<p><tr><td>Build URL:&nbsp;</td><td><a href={args.url}> {args.url} </a></td></tr></p>
    <p>find perf regression or improvement from attachment report, Thanks</p>
  </div>
  </div>
  </div>
  </div>
<script src="js/jquery-3.2.1.min.js"></script>
<script src="js/popper.js"></script>
<script src="js/bootstrap.min.js"></script>
<script src="js/select2/select2.min.js"></script>
<script src="js/main.js"></script>
</body>'''

def html_generate(html_off):
    if not html_off:
        try:
            content = pd.read_excel(args.target+'/inductor_log/Inductor Dashboard Regression Check '+args.target+'.xlsx',sheet_name=[0,1,2,3])
            summary= pd.DataFrame(content[0]).to_html(classes="table",index = False)
            swinfo= pd.DataFrame(content[1]).to_html(classes="table",index = False)
            mt_failures= pd.DataFrame(content[2]).to_html(classes="table",index = False)
            st_failures= pd.DataFrame(content[3]).to_html(classes="table",index = False)
            with open(args.target+'/inductor_log/inductor_model_bench.html',mode = "a") as f:
                f.write(html_head()+"<p>Summary</p>"+summary+"<p>SW info</p>"+swinfo+"<p>Multi-threads Failures</p>"+mt_failures+"<p>Single-thread Failures</p>"+st_failures+html_tail())
            f.close()
        except:
            print("html_generate_failed")
            pass

def generate_report(excel,reference,target):
    update_summary(excel,reference,target)
    update_swinfo(excel)
    if args.mode == 'multiple' or args.mode == 'all':
        update_failures(excel,target_mt)
    if args.mode =='single' or args.mode == 'all':
        update_failures(excel,target_st)
    update_details(excel)

def excel_postprocess(file):
    wb=file.book
    # Summary
    ws=wb['Summary']
    if args.reference is not None:
        ws.merge_cells(start_row=2,end_row=5,start_column=1,end_column=1)
        ws.merge_cells(start_row=6,end_row=9,start_column=1,end_column=1)
        ws.merge_cells(start_row=2,end_row=3,start_column=2,end_column=2)
        ws.merge_cells(start_row=4,end_row=5,start_column=2,end_column=2)
        ws.merge_cells(start_row=6,end_row=7,start_column=2,end_column=2)
        ws.merge_cells(start_row=8,end_row=9,start_column=2,end_column=2)
    else:
        ws.merge_cells(start_row=2,end_row=3,start_column=1,end_column=1)
        ws.merge_cells(start_row=4,end_row=5,start_column=1,end_column=1)
    if args.mode == "multiple" or args.mode == 'all':  
        # Single-Socket Multi-threads
        wmt=wb['Single-Socket Multi-threads']
        wmt.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=torchbench_index+3,end_row=hf_index+2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=hf_index+3,end_row=timm_index+2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=timm_index+3,end_row=end_index+3,start_column=1,end_column=1)
        wmt.merge_cells(start_row=1,end_row=1,start_column=3,end_column=6)
        wmt.merge_cells(start_row=1,end_row=1,start_column=7,end_column=10)
        wmt.merge_cells(start_row=1,end_row=1,start_column=11,end_column=13)
    if args.mode == "single" or args.mode == 'all':
        # Single-Core Single-thread
        wst=wb['Single-Core Single-thread']
        wst.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
        wst.merge_cells(start_row=torchbench_index+3,end_row=hf_index+2,start_column=1,end_column=1)
        wst.merge_cells(start_row=hf_index+3,end_row=timm_index+2,start_column=1,end_column=1)
        wst.merge_cells(start_row=timm_index+3,end_row=end_index+3,start_column=1,end_column=1)    
        wst.merge_cells(start_row=1,end_row=1,start_column=3,end_column=6)
        wst.merge_cells(start_row=1,end_row=1,start_column=7,end_column=10)
        wst.merge_cells(start_row=1,end_row=1,start_column=11,end_column=13)
    wb.save(file)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter(args.target+'/inductor_log/Inductor Dashboard Regression Check '+args.target+'.xlsx')
    generate_report(excel,args.reference, args.target)
    excel_postprocess(excel)
    html_generate(args.html_off)
    update_issue_commits()