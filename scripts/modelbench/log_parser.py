"""log_parser.py
Generate report from two specified inductor logs.
Usage:
  python log_parser.py --reference WW48.2 --target WW48.4

"""


import argparse
import pandas as pd
from pandas import ExcelWriter
import os

parser = argparse.ArgumentParser(description="Generate report from two specified inductor logs")
parser.add_argument('-t','--target',type=str,help='target log file')
parser.add_argument('-r','--reference',type=str,help='reference log file')
args=parser.parse_args()

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                return os.path.join(root, d)
        for f in files:
            if thread in (os.path.join(root, f)):
                return os.path.join(root, f)                 

reference_mt=getfolder(args.reference,'multi_threads_cf_logs')
reference_st=getfolder(args.reference,'single_thread_cf_logs')
target_mt=getfolder(args.target,'multi_threads_cf_logs')
target_st=getfolder(args.target,'single_thread_cf_logs')

def update_summary(writer,reference,target):
    data = {
        'Test Secnario':['Single Socket Multi-Threads', ' ', ' ', ' ','Single Core Single-Thread',' ',' ',' '], 
        'Comp Item':['Pass Rate', ' ', 'Geomean Speedup', ' ','Pass Rate',' ','Geomean Speedup',' '],
        'Date':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor','inductor','inductor','inductor'],
        'torchbench':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'huggingface':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
        'timm_models ':[' ', ' ', ' ', ' ',' ',' ',' ',' ']
    }
    summary=pd.DataFrame(data)
    # read reference round test results
    reference_mt_pr_data=pd.read_csv(reference_mt+'/passrate.csv',index_col=0)
    reference_mt_gm_data=pd.read_csv(reference_mt+'/geomean.csv',index_col=0)
    reference_st_pr_data=pd.read_csv(reference_st+'/passrate.csv',index_col=0)
    reference_st_gm_data=pd.read_csv(reference_st+'/geomean.csv',index_col=0)
    # update
    summary.iloc[0:1,4:7]=reference_mt_pr_data.iloc[0:2,1:7]
    summary.iloc[2:3,4:7]=reference_mt_gm_data.iloc[0:2,1:7]
    summary.iloc[4:5,4:7]=reference_st_pr_data.iloc[0:2,1:7]
    summary.iloc[6:7,4:7]=reference_st_gm_data.iloc[0:2,1:7]

    summary.iloc[0:1,2]=reference
    summary.iloc[2:3,2]=reference
    summary.iloc[4:5,2]=reference
    summary.iloc[6:7,2]=reference
    # read target round test results
    target_mt_pr_data=pd.read_csv(target_mt+'/passrate.csv',index_col=0)
    target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0)
    target_st_pr_data=pd.read_csv(target_st+'/passrate.csv',index_col=0)
    target_st_gm_data=pd.read_csv(target_st+'/geomean.csv',index_col=0)
    # update
    summary.iloc[1:2,4:7]=target_mt_pr_data.iloc[0:2,1:7]
    summary.iloc[3:4,4:7]=target_mt_gm_data.iloc[0:2,1:7]
    summary.iloc[5:6,4:7]=target_st_pr_data.iloc[0:2,1:7]
    summary.iloc[7:8,4:7]=target_st_gm_data.iloc[0:2,1:7]

    summary.iloc[1:2,2]=target
    summary.iloc[3:4,2]=target
    summary.iloc[5:6,2]=target
    summary.iloc[7:8,2]=target

    summary.to_excel(writer,sheet_name='Summary', index=False)  

def update_swinfo(writer):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','dynamo/benchmarks'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' '],'Master/Main commit':[' ', ' ', ' ', ' ',' ',' ']}
    swinfo=pd.DataFrame(data)
    swinfo.to_excel(writer, sheet_name='SW', index=False)

def update_failures(writer):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
        acc_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_accuracy.csv'

        perf_data=pd.read_csv(perf_path,index_col=0)
        acc_data=pd.read_csv(acc_path,index_col=0)
        
        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['accuracy'] ==0),:]
        tmp.append(acc_data)

        perf_data=perf_data.loc[perf_data['speedup'] ==0,:]
        tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['name','accuracy','speedup']]

    failures['accuracy'].replace(["fail_to_run","fail_accuracy","0"],["X","X","X"],inplace=True)
    failures['speedup'].replace([0],["X"],inplace=True)
    
    failures.to_excel(writer, sheet_name='Failures', index=False)


def process_suite(suite,thread):
    reference_file_path=getfolder(args.reference,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
    target_file_path=getfolder(args.target,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'

    reference_ori_data=pd.read_csv(reference_file_path,index_col=0)
    target_ori_data=pd.read_csv(target_file_path,index_col=0)

    reference_data=reference_ori_data[['name','batch_size','speedup']]
    target_data=target_ori_data[['name','batch_size','speedup']]

    reference_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)
    
    data=pd.merge(target_data,reference_data,on=['name'],how= 'outer')
    return data

def process_thread(thread):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        data=process_suite(suite, thread)
        tmp.append(data)
    return pd.concat(tmp)
 
all_models = []

def parse_log(file):
    result = []
    suite = []
    with open(file, 'r') as reader:
        contents = reader.readlines()
        model = ""
        for line in contents:
            if "Time cost" in line:
                model = line.split(" Time cost")[0].split(" ")[-1].strip()
                if model not in suite:
                    suite.append(model)
            elif line.startswith("eager: "):
                result.append(model+", "+ line)
            elif "cpu  eval" in line:
                m = line.split("cpu  eval")[-1].strip().split(" ")[0].strip()
                if m not in suite:
                    suite.append(m)
            elif line.startswith("compression_ratio"):
                suite.sort(key=str.lower)
                all_models.extend(suite)
                suite.clear()
    return result

def str_to_dict(contents):
    res_dict = {}
    for line in contents:
        model = line.split(",")[0]
        eager = float(line.split(",")[1].strip().split(":")[-1])
        inductor = float(line.split(",")[2].strip().split(":")[-1])
        res_dict[model] = [eager, inductor]
    return res_dict

def process_absolute_data(thread):
    target_log=getfolder(args.target,thread)
    reference_log=getfolder(args.reference,thread)
    new_res = parse_log(target_log)
    old_res = parse_log(reference_log)
    new_res_dict = str_to_dict(new_res)
    old_res_dict = str_to_dict(old_res)
    results = ["name, Eager(new), Inductor(new), Eager(old), Inductor(old), Eager Ratio(old/new), Inductor Ratio(old/new)\n"]

    unique_models = []
    for item in all_models:
        if item not in unique_models:
            unique_models.append(item)

    for key in unique_models:
        line = key+", "
        if key in new_res_dict:
            for item in new_res_dict[key]:
                line += str(item) +", "
        else:
            line += "NA, NA, "
        if key in old_res_dict:
            for item in old_res_dict[key]:
                line += str(item) +", "
        else:
            line += "NA, NA, "
        if key in old_res_dict and key in new_res_dict:
            line+=str(round(old_res_dict[key][0]/new_res_dict[key][0], 2)) + ", "
            line+=str(round(old_res_dict[key][1]/new_res_dict[key][1], 2))
        else:
            line += "NA, NA, "
        line += "\n"
        results.append(line)
    return results

def update_details(writer):
    header = {"A": '', "B": args.target, "C": '', "D": '',"E": '', "F": args.reference, "G": '', "H": '',"I": '',"J": 'Result Comp',"K": '',"L": ''}
    h = pd.DataFrame(header, index=[0])
    h.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
    h.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False,startrow=0,header=False)
    
    # update cmp data
    mt=process_thread('multi_threads_cf_logs')
    st=process_thread('single_thread_cf_logs')

    mt_old=mt[['name','batch_size_x','speedup_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new'})
    mt_new=mt[['batch_size_y','speedup_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old'})

    mt_old.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=0)
    mt_new.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=5)

    st_old=st[['name','batch_size_x','speedup_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new'})
    st_new=st[['batch_size_y','speedup_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old'})    

    st_old.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=0)
    st_new.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=5)


    mt_ratio = pd.DataFrame(mt['speedup_x'] / mt['speedup_y'],columns=['Ratio Speedup(New/old)'])
    st_ratio = pd.DataFrame(st['speedup_x'] / st['speedup_y'],columns=['Ratio Speedup(New/old)'])

    mt_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=9)
    st_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=9)

    # update abs data
    mt_abs_data=process_absolute_data('multi_threads_model_bench_log')
    st_abs_data=process_absolute_data('single_thread_model_bench_log')
    
    mt_abs=pd.DataFrame(mt_abs_data)
    st_abs=pd.DataFrame(st_abs_data)
    
    m_abs=pd.DataFrame(mt_abs[0].str.split(', ',expand=True))
    s_abs=pd.DataFrame(st_abs[0].str.split(', ',expand=True))

    m_abs[[1,2]].to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=False, startrow=1, startcol=3)
    m_abs[[3,4]].to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=False, startrow=1, startcol=7)
    m_abs[[5,6]].to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=False, startrow=1, startcol=10)

    s_abs[[1,2]].to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=False, startrow=1, startcol=3)
    s_abs[[3,4]].to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=False, startrow=1, startcol=7)
    s_abs[[5,6]].to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=False, startrow=1, startcol=10)

def generate_report(reference,target):
    with ExcelWriter('Inductor Dashboard Regression Check '+target+'.xlsx') as writer:
        update_summary(writer,reference,target)
        update_swinfo(writer)
        update_failures(writer)
        update_details(writer)

if __name__ == '__main__':
    generate_report(args.reference, args.target)
