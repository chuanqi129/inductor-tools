"""process.py
Generate a report for review from specified two directories of test results.
Usage:
  python process.py --lr WW48.2 --tr WW48.4

  lr: last round
  tr: this round

"""


import argparse
import pandas as pd
from pandas import ExcelWriter
import os

parser = argparse.ArgumentParser(description="Generate report from recently two round inductor_log")
parser.add_argument('-l','--lr',type=str,help='last round log file')
parser.add_argument('-t','--tr',type=str,help='this round log file')
args=parser.parse_args()

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                return os.path.join(root, d)
        for f in files:
            if thread in (os.path.join(root, f)):
                return os.path.join(root, f)                 

lr_mt=getfolder(args.lr,'multi_threads_cf_logs')
lr_st=getfolder(args.lr,'single_thread_cf_logs')
tr_mt=getfolder(args.tr,'multi_threads_cf_logs')
tr_st=getfolder(args.tr,'single_thread_cf_logs')

def update_summary(writer,lr,tr):
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
    # read last round test results
    lr_mt_pr_data=pd.read_csv(lr_mt+'/passrate.csv',index_col=0)
    lr_mt_gm_data=pd.read_csv(lr_mt+'/geomean.csv',index_col=0)
    lr_st_pr_data=pd.read_csv(lr_st+'/passrate.csv',index_col=0)
    lr_st_gm_data=pd.read_csv(lr_st+'/geomean.csv',index_col=0)
    # update
    summary.iloc[0:1,4:7]=lr_mt_pr_data.iloc[0:2,1:7]
    summary.iloc[2:3,4:7]=lr_mt_gm_data.iloc[0:2,1:7]
    summary.iloc[4:5,4:7]=lr_st_pr_data.iloc[0:2,1:7]
    summary.iloc[6:7,4:7]=lr_st_gm_data.iloc[0:2,1:7]

    summary.iloc[0:1,2]=lr
    summary.iloc[2:3,2]=lr
    summary.iloc[4:5,2]=lr
    summary.iloc[6:7,2]=lr
    # read this round test results
    tr_mt_pr_data=pd.read_csv(tr_mt+'/passrate.csv',index_col=0)
    tr_mt_gm_data=pd.read_csv(tr_mt+'/geomean.csv',index_col=0)
    tr_st_pr_data=pd.read_csv(tr_st+'/passrate.csv',index_col=0)
    tr_st_gm_data=pd.read_csv(tr_st+'/geomean.csv',index_col=0)
    # update
    summary.iloc[1:2,4:7]=tr_mt_pr_data.iloc[0:2,1:7]
    summary.iloc[3:4,4:7]=tr_mt_gm_data.iloc[0:2,1:7]
    summary.iloc[5:6,4:7]=tr_st_pr_data.iloc[0:2,1:7]
    summary.iloc[7:8,4:7]=tr_st_gm_data.iloc[0:2,1:7]

    summary.iloc[1:2,2]=tr
    summary.iloc[3:4,2]=tr
    summary.iloc[5:6,2]=tr
    summary.iloc[7:8,2]=tr

    summary.to_excel(writer,sheet_name='Summary', index=False)  

def update_swinfo(writer):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','dynamo/benchmarks'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' '],'Master/Main commit':[' ', ' ', ' ', ' ',' ',' ']}
    swinfo=pd.DataFrame(data)
    swinfo.to_excel(writer, sheet_name='SW', index=False)

def update_failures(writer):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=tr_mt+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
        acc_path=tr_mt+'/inductor_'+suite+'_float32_inference_cpu_accuracy.csv'

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
    lr_file_path=getfolder(args.lr,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
    tr_file_path=getfolder(args.tr,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'

    lr_ori_data=pd.read_csv(lr_file_path,index_col=0)
    tr_ori_data=pd.read_csv(tr_file_path,index_col=0)

    lr_data=lr_ori_data[['name','batch_size','speedup']]
    tr_data=tr_ori_data[['name','batch_size','speedup']]

    lr_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)
    tr_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)
    
    data=pd.merge(tr_data,lr_data,on=['name'],how= 'outer')
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
    tr_log=getfolder(args.tr,thread)
    lr_log=getfolder(args.lr,thread)
    new_res = parse_log(tr_log)
    old_res = parse_log(lr_log)
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
    header = {"A": '', "B": args.tr, "C": '', "D": '',"E": '', "F": args.lr, "G": '', "H": '',"I": '',"J": 'Result Comp',"K": '',"L": ''}
    h = pd.DataFrame(header, index=[0])
    h.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
    h.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False,startrow=0,header=False)
    
    # update cmp data
    mt=process_thread('multi_threads_cf_logs')
    st=process_thread('single_thread_cf_logs')

    mt[['name','batch_size_x','speedup_x']].to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=0)
    mt[['batch_size_y','speedup_y']].to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=5)

    st[['name','batch_size_x','speedup_x']].to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=0)
    st[['batch_size_y','speedup_y']].to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=5)         

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

def generate_report(lr,tr):
    with ExcelWriter('Inductor Dashboard Regression Check '+tr+'.xlsx') as writer:
        update_summary(writer,lr,tr)
        update_swinfo(writer)
        update_failures(writer)
        update_details(writer)

if __name__ == '__main__':
    generate_report(args.lr, args.tr)