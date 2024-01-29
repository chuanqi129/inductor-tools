"""
Generate report or data compare report from specified inductor logs.
Usage:
  python report.py -r WW48.2 -t WW48.4 -m all --html_off --md_off --precision bfloat16
  python report.py -r WW48.2 -t WW48.4 -m all --gh_token github_pat_xxxxx --dashboard dynamic
Dependencies:
    styleframe
    PyGithub
    datacompy
"""

import argparse
import datacompy
from datetime import datetime,timedelta
from scipy.stats import gmean
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
parser.add_argument('--cppwrapper_gm', action='store_true',help='turn on geomean speedup calculation on cppwrapper vs pythonwrapper')
parser.add_argument('--mt_interval_start', type=float,default=0.04,help='cppwrapper gm mt interval start')
parser.add_argument('--mt_interval_end', type=float,default=1.5,help='cppwrapper gm mt interval end')
parser.add_argument('--st_interval_start', type=float,default=0.04,help='cppwrapper sm mt interval start')
parser.add_argument('--st_interval_end', type=float,default=5,help='cppwrapper gm st interval end')
parser.add_argument('--image_tag', type=str,help='image tag which used in tests')
parser.add_argument('--suite',type=str,default='all',help='Test suite: torchbench, huggingface, timm_models')
parser.add_argument('--infer_or_train',type=str,default='inference',help='inference or training')
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

new_performance_regression=pd.DataFrame()
new_failures=pd.DataFrame()
new_performance_improvement=pd.DataFrame()
new_fixed_failures=pd.DataFrame()

# cppwrapper gm values
multi_threads_gm={}
single_thread_gm={}

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                return os.path.join(root, d)
        for f in files:
            if thread in (os.path.join(root, f)):
                return os.path.join(root, f)                 
reference_mt=''
reference_st=''
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

if args.suite == "all":
    suite_list = ['torchbench','huggingface','timm_models']
else:
    suite_list = [args.suite]

def filter_df_by_threshold(df, interval_start,interval_end):
    df = df[df['inductor_new'] > 0]
    filtered_data_small = df[df['inductor_old'] <= interval_start]
    filtered_data_medium = df[(df['inductor_old'] > interval_start)  & (df['inductor_old'] <= interval_end)]
    filtered_data_large = df[df['inductor_old'] > interval_end]
    return filtered_data_small, filtered_data_medium,filtered_data_large

def get_passing_entries(df,column_name):
    return df[column_name][df[column_name] > 0]

def caculate_geomean(df,column_name):
    cleaned_df = get_passing_entries(df,column_name).clip(1)
    if cleaned_df.empty:
        return "0.0x"
    return f"{gmean(cleaned_df):.2f}x"

def update_summary(excel, reference, target):
    if args.suite == 'all':
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
    else:
        data = {
            'Test Secnario':['Single Socket Multi-Threads', ' ', ' ', ' ','Single Core Single-Thread',' ',' ',' '], 
            'Comp Item':['Pass Rate', ' ', 'Geomean Speedup', ' ','Pass Rate',' ','Geomean Speedup',' '],
            'Date':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor','inductor','inductor','inductor'],
            args.suite:[' ', ' ', ' ', ' ',' ',' ',' ',' ']
        }
        data_target = {
            'Test Secnario':['Single Socket Multi-Threads', ' ','Single Core Single-Thread',' '], 
            'Comp Item':['Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup'],
            'Date':[' ', ' ', ' ', ' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor'],
            args.suite:[' ', ' ', ' ', ' ']
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
    for i in range(1, len(data)+1):
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

def get_failures(target_path):
    tmp=[]
    for suite_name in suite_list:
        perf_path = '{0}/inductor_{1}_{2}_{3}_cpu_performance.csv'.format(target_path, suite_name, args.precision, args.infer_or_train)
        acc_path = '{0}/inductor_{1}_{2}_{3}_cpu_accuracy.csv'.format(target_path, suite_name, args.precision, args.infer_or_train)

        perf_data=pd.read_csv(perf_path)
        acc_data=pd.read_csv(acc_path)

        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='infra_error') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
        acc_data.insert(loc=2, column='acc_suite', value=suite_name)
        tmp.append(acc_data)

        perf_data=perf_data.loc[(perf_data['batch_size'] ==0) | (perf_data['speedup'] ==0) | (perf_data['speedup'] =='infra_error'),:]
        perf_data.insert(loc=3, column='pef_suite', value=suite_name)
        tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['acc_suite','pef_suite','name','accuracy','speedup']]
    failures['pef_suite'].fillna(0,inplace=True)
    failures['acc_suite'].fillna(0,inplace=True)
    # There is no failure in accuracy and performance, just return
    if (len(failures['acc_suite']) == 0) and (len(failures['pef_suite']) == 0):
        return failures
    failures['suite'] = failures.apply(lambda x: x['pef_suite'] if x['acc_suite']==0 else x['acc_suite'], axis=1) 
    failures=failures.rename(columns={'suite':'suite','name':'name','accuracy':'accuracy','speedup':'perf'}) 

    # 1 -> failed
    failures['accuracy'].replace(['infra_error','timeout','fail_to_run','fail_accuracy','0.0000','model_fail_to_load','eager_fail_to_run'],[1,1,1,1,1,1,1],inplace=True)
    failures['perf'].replace([0],['fail'],inplace=True)
    failures['perf'].replace(['fail','infra_error','timeout'],[1,1,1],inplace=True)
    failures['suite'].replace(["torchbench","huggingface","timm_models"],[3,4,5],inplace=True)   
    failures=failures.groupby(by=['name']).sum(numeric_only=True).reset_index()
    failures['suite'].replace([3,4,5,6,8,10],["torchbench","huggingface","timm_models","torchbench","huggingface","timm_models"],inplace=True)
    failures['perf'].replace([0,1],["√","X"],inplace=True)
    failures['accuracy'].replace([0,1],["√","X"],inplace=True)  
    # fill failure reasons
    failures['name']=failures['name'].drop_duplicates()
    failures_dict =  {key:values for key, values in zip(failures['name'], failures['accuracy'])}
    reason_content=[]
    for model in failures_dict.keys():
        reason_content.append(failures_reason_parse(model,failures_dict[model],target_path))
    failures['reason(reference only)']=reason_content        
    return failures

def update_failures(excel,target_thread,refer_thread):
    global new_failures
    global new_fixed_failures
    target_thread_failures = get_failures(target_thread)    
    # new failures compare with reference logs
    if args.reference is not None:
        refer_thread_failures = get_failures(refer_thread)
        # New Failures
        failure_regression_compare = datacompy.Compare(target_thread_failures, refer_thread_failures, join_columns='name')
        failure_regression = failure_regression_compare.df1_unq_rows.copy()
        new_failures = pd.concat([new_failures,failure_regression])

        # Fixed Failures
        fixed_failures_compare = datacompy.Compare(refer_thread_failures, target_thread_failures, join_columns='name')
        fixed_failures = fixed_failures_compare.df1_unq_rows.copy()
        new_fixed_failures = pd.concat([new_fixed_failures,fixed_failures])

    # There is no failure in target, just return
    if (len(target_thread_failures) == 0):
        return
    sf = StyleFrame({'suite': list(target_thread_failures['suite']),
                 'name': list(target_thread_failures['name']),
                 'accuracy': list(target_thread_failures['accuracy']),
                 'perf': list(target_thread_failures['perf']),
                 'reason(reference only)':(list(target_thread_failures['reason(reference only)']))})
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
    if args.reference is not None:    
        new_failures_list = new_failures['name'].values.tolist()
        for failed_model in new_failures_list:
            sf.apply_style_by_indexes(indexes_to_style=sf[sf['name'] == failed_model],styler_obj=regression_style)
    sf.to_excel(sheet_name='Failures in '+target_thread.split('_cf')[0].split('inductor_log/')[1].strip(),excel_writer=excel,index=False)

def process_suite(suite,thread):
    target_file_path = '{0}/inductor_{1}_{2}_{3}_cpu_performance.csv'.format(getfolder(args.target, thread), suite, args.precision, args.infer_or_train)
    target_ori_data=pd.read_csv(target_file_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency','compilation_latency']]
    target_data=target_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)

    if args.reference is not None:
        reference_file_path = '{0}/inductor_{1}_{2}_{3}_cpu_performance.csv'.format(getfolder(args.reference, thread), suite, args.precision, args.infer_or_train)
        reference_ori_data=pd.read_csv(reference_file_path,index_col=0)
        reference_data=reference_ori_data[['name','batch_size','speedup','abs_latency','compilation_latency']]
        reference_data=reference_data.copy()
        reference_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)    
        data=pd.merge(target_data,reference_data,on=['name'],how= 'outer')
        data['suite'] = suite
        return data
    else:
        target_data['suite'] = suite
        return target_data

def process_thread(thread):
    tmp=[]
    for suite in suite_list:
        data=process_suite(suite, thread)
        tmp.append(data)
    return pd.concat(tmp)
 
def process(input,thread):
    if args.reference is not None:
        data_new=input[['name','suite','batch_size_x','speedup_x','abs_latency_x','compilation_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new',"compilation_latency_x":'compilation_latency_new'})
        data_new['inductor_new']=data_new['inductor_new'].astype(float).div(1000)
        data_new['speed_up_new']=data_new['speed_up_new'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager_new'] = data_new['speed_up_new'] * data_new['inductor_new']        
        data_old=input[['batch_size_y','speedup_y','abs_latency_y','compilation_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old',"compilation_latency_y":'compilation_latency_old'})    
        data_old['inductor_old']=data_old['inductor_old'].astype(float).div(1000)
        data_old['speed_up_old']=data_old['speed_up_old'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_old['eager_old'] = data_old['speed_up_old'] * data_old['inductor_old']
        input['speedup_x']=input['speedup_x'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        input['speedup_y']=input['speedup_y'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_ratio= pd.DataFrame(round(input['speedup_x'] / input['speedup_y'],2),columns=['Ratio Speedup(New/old)'])
        data_ratio['Eager Ratio(old/new)'] = pd.DataFrame(round(data_old['eager_old'] / data_new['eager_new'],2))
        data_ratio['Inductor Ratio(old/new)'] = pd.DataFrame(round(data_old['inductor_old'] / data_new['inductor_new'],2))
        data_ratio['Compilation_latency_Ratio(old/new)'] = pd.DataFrame(round(data_old['compilation_latency_old'] / data_new['compilation_latency_new'],2))
        
        combined_data = pd.DataFrame({
            'name': list(data_new['name']),
            'suite': list(data_new['suite']),
            'batch_size_new': list(data_new['batch_size_new']),
            'speed_up_new': list(data_new['speed_up_new']),
            'inductor_new': list(data_new['inductor_new']),
            'eager_new': list(data_new['eager_new']),
            'compilation_latency_new': list(data_new['compilation_latency_new']),
            'batch_size_old': list(data_old['batch_size_old']),
            'speed_up_old': list(data_old['speed_up_old']),
            'inductor_old': list(data_old['inductor_old']),
            'eager_old': list(data_old['eager_old']),
            'compilation_latency_old': list(data_old['compilation_latency_old']),
            'Ratio Speedup(New/old)': list(data_ratio['Ratio Speedup(New/old)']),
            'Eager Ratio(old/new)': list(data_ratio['Eager Ratio(old/new)']),
            'Inductor Ratio(old/new)': list(data_ratio['Inductor Ratio(old/new)']),
            'Compilation_latency_Ratio(old/new)': list(data_ratio['Compilation_latency_Ratio(old/new)'])
            })
        if args.cppwrapper_gm:
            global multi_threads_gm, single_thread_gm
            if thread == "multiple":
                combined_data_small, combined_data_medium,combined_data_large=filter_df_by_threshold(combined_data,args.mt_interval_start,args.mt_interval_end)             
                multi_threads_gm['small'] = caculate_geomean(combined_data_small,'Inductor Ratio(old/new)')
                multi_threads_gm['medium'] = caculate_geomean(combined_data_medium,'Inductor Ratio(old/new)')
                multi_threads_gm['large'] = caculate_geomean(combined_data_large,'Inductor Ratio(old/new)')
            if thread == "single":
                combined_data_small, combined_data_medium,combined_data_large=filter_df_by_threshold(combined_data,args.st_interval_start,args.st_interval_end)
                single_thread_gm['small'] = caculate_geomean(combined_data_small,'Inductor Ratio(old/new)')
                single_thread_gm['medium'] = caculate_geomean(combined_data_medium,'Inductor Ratio(old/new)')
                single_thread_gm['large'] = caculate_geomean(combined_data_large,'Inductor Ratio(old/new)')      
        data = StyleFrame(combined_data)
        data.set_column_width(1, 10)
        data.set_column_width(2, 10)
        data.set_column_width(3, 18)
        data.set_column_width(4, 18)
        data.set_column_width(5, 18)
        data.set_column_width(6, 15)
        data.set_column_width(7, 20)
        data.set_column_width(8, 18)
        data.set_column_width(9, 18)
        data.set_column_width(10, 18)
        data.set_column_width(11, 15)
        data.set_column_width(12, 20)
        data.set_column_width(13, 28)
        data.set_column_width(14, 28)
        data.set_column_width(15, 28)
        data.set_column_width(16, 32)
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size_new'] == 0], styler_obj=red_style)
        data.apply_style_by_indexes(indexes_to_style=data[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < 0.9)],styler_obj=regression_style)
        global new_performance_regression
        regression = data.loc[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < 0.9)]
        regression = regression.copy()
        new_performance_regression = pd.concat([new_performance_regression,regression])
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] > 1.1],styler_obj=improve_style)
        data.set_row_height(rows=data.row_indexes, height=15)

        global new_performance_improvement
        improvement = data.loc[(data['Inductor Ratio(old/new)'] > 1.1)]
        improvement = improvement.copy()
        new_performance_improvement = pd.concat([new_performance_improvement, improvement])
    else:
        data_new=input[['name','suite','batch_size','speedup','abs_latency','compilation_latency']].rename(columns={'name':'name','batch_size':'batch_size','speedup':'speedup',"abs_latency":'inductor',"compilation_latency":'compilation_latency'})
        data_new['inductor']=data_new['inductor'].astype(float).div(1000)
        data_new['speedup']=data_new['speedup'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager'] = data_new['speedup'] * data_new['inductor']        
        data = StyleFrame({
            'name': list(data_new['name']),
            'suite': list(data_new['suite']),
            'batch_size': list(data_new['batch_size']),
            'speedup': list(data_new['speedup']),
            'inductor': list(data_new['inductor']),
            'eager': list(data_new['eager']),
            'compilation_latency': list(data_new['compilation_latency']),})
        data.set_column_width(1, 10)
        data.set_column_width(2, 10)
        data.set_column_width(3, 18)
        data.set_column_width(4, 18)
        data.set_column_width(5, 18)
        data.set_column_width(6, 15)
        data.set_column_width(7, 20)
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size'] == 0], styler_obj=red_style)
        data.set_row_height(rows=data.row_indexes, height=15)
    return data

def update_details(writer):
    h = {"A": 'Model', "B": 'Suite', "C": args.target, "D": '', "E": '',"F": '', "G": '',"H": args.reference, "I": '', "J": '',"K": '',"L":'',"M": 'Result Comp',"N": '',"O": '',"P":''}
    if args.reference is None:
        h = {"A": 'Model',"B": 'Suite', "C": args.target, "D": '', "E": '',"F": '',"G":''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    if args.mode == "multiple" or args.mode == 'all':
        # mt
        head.to_excel(excel_writer=writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
        mt=process_thread('multi_threads_cf_logs')
        mt_data=process(mt,"multiple")
        mt_data.to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=0)    
    if args.mode == "single" or args.mode == 'all':
        # st
        head.to_excel(excel_writer=writer, sheet_name='Single-Core Single-thread', index=False,startrow=0,header=False) 
        st=process_thread('single_thread_cf_logs')
        st_data=process(st,"single")
        st_data.to_excel(sheet_name='Single-Core Single-thread',excel_writer=writer,index=False,startrow=1,startcol=0)

def generate_report(excel, reference, target):
    update_summary(excel, reference, target)
    update_swinfo(excel)
    if args.mode == 'multiple' or args.mode == 'all':
        update_failures(excel,target_mt,reference_mt)
    if args.mode =='single' or args.mode == 'all':
        update_failures(excel,target_st,reference_st)
    update_details(excel)
    if args.cppwrapper_gm:
        update_cppwrapper_gm(excel,reference,target)

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
        wmt.merge_cells(start_row=1,end_row=2,start_column=2,end_column=2)
        wmt.merge_cells(start_row=1,end_row=1,start_column=3,end_column=7)
        wmt.merge_cells(start_row=1,end_row=1,start_column=8,end_column=12)
        wmt.merge_cells(start_row=1,end_row=1,start_column=13,end_column=16)
    if args.mode == "single" or args.mode == 'all':
        # Single-Core Single-thread
        wst=wb['Single-Core Single-thread']
        wst.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=1,end_row=2,start_column=2,end_column=2)  
        wst.merge_cells(start_row=1,end_row=1,start_column=3,end_column=7)
        wst.merge_cells(start_row=1,end_row=1,start_column=8,end_column=12)
        wst.merge_cells(start_row=1,end_row=1,start_column=13,end_column=16)
    wb.save(file)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter('{0}/inductor_log/Inductor Dashboard Regression Check {0} {1}.xlsx'.format(args.target, args.suite))
    generate_report(excel, args.reference, args.target)
    excel_postprocess(excel)
    html_generate(args.html_off)     
    update_issue_commits(args.precision)