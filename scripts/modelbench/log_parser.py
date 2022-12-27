"""log_parser.py
Generate report from two specified inductor logs.
Usage:
  python log_parser.py --reference WW48.2 --target WW48.4

"""
# Not ready for updated benchmarks for nightly

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
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' '],'Master/Main commit':[' ', ' ', ' ', ' ',' ',' ']}
    swinfo=pd.DataFrame(data)
    swinfo.to_excel(writer, sheet_name='SW', index=False)

def update_failures(writer):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
        acc_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_accuracy.csv'

        perf_data=pd.read_csv(perf_path,index_col=0)
        acc_data=pd.read_csv(acc_path,index_col=0)
        
        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
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

    reference_data=reference_ori_data[['name','batch_size','speedup','abs_latency']]
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency']]

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
 

def update_details(writer):
    header = {"A": '', "B": args.target, "C": '', "D": '',"E": '', "F": args.reference, "G": '', "H": '',"I": '',"J": 'Result Comp',"K": '',"L": ''}
    h = pd.DataFrame(header, index=[0])
    h.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
    h.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False,startrow=0,header=False)
    
    # update cmp data
    mt=process_thread('multi_threads_cf_logs')
    st=process_thread('single_thread_cf_logs')

    mt_new=mt[['name','batch_size_x','speedup_x','abs_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new'})
    mt_old=mt[['batch_size_y','speedup_y','abs_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old'})
    mt_new['inductor_new']=mt_new['inductor_new'].astype(float).div(1000)
    mt_old['inductor_old']=mt_old['inductor_old'].astype(float).div(1000)
    mt_new.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=0)
    mt_old.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, startrow=1, startcol=5)

    st_new=st[['name','batch_size_x','speedup_x','abs_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new'})
    st_old=st[['batch_size_y','speedup_y','abs_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old'})    
    st_new['inductor_new']=st_new['inductor_new'].astype(float).div(1000)
    st_old['inductor_old']=st_old['inductor_old'].astype(float).div(1000)
    st_new.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=0)
    st_old.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, startrow=1, startcol=5)

    mt_eager_new = pd.DataFrame(mt_new['speed_up_new'] * mt_new['inductor_new'],columns=['eager_new'])
    mt_eager_old = pd.DataFrame(mt_old['speed_up_old'] * mt_old['inductor_old'],columns=['eager_old'])
    st_eager_new = pd.DataFrame(st_new['speed_up_new'] * st_new['inductor_new'],columns=['eager_new'])
    st_eager_old = pd.DataFrame(st_old['speed_up_old'] * st_old['inductor_old'],columns=['eager_old'])    

    mt_ratio = pd.DataFrame(mt['speedup_x'] / mt['speedup_y'],columns=['Ratio Speedup(New/old)'])
    st_ratio = pd.DataFrame(st['speedup_x'] / st['speedup_y'],columns=['Ratio Speedup(New/old)'])

    mt_eager_ratio = pd.DataFrame(mt_eager_old['eager_old'] / mt_eager_new['eager_new'],columns=['Eager Ratio(old/new)'])
    mt_inductor_ratio = pd.DataFrame(mt_old['inductor_old'] / mt_new['inductor_new'],columns=['Inductor Ratio(old/new)'])
    st_eager_ratio = pd.DataFrame(st_eager_old['eager_old'] / st_eager_new['eager_new'],columns=['Eager Ratio(old/new)'])   
    st_inductor_ratio = pd.DataFrame(st_old['inductor_old'] / st_new['inductor_new'],columns=['Inductor Ratio(old/new)'])    

    mt_eager_new.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=4)
    mt_eager_old.to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=8)
    mt_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=9)
    mt_eager_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=10) 
    mt_inductor_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Multi-threads', index=False, header=True, startrow=1, startcol=11)   
     
    st_eager_new.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=4)
    st_eager_old.to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=8)      
    st_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=9)
    st_eager_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=10) 
    st_inductor_ratio.round(2).to_excel(writer, sheet_name='Single-Socket Single-thread', index=False, header=True, startrow=1, startcol=11)     

def generate_report(reference,target):
    with ExcelWriter('Inductor Dashboard Regression Check '+target+'.xlsx') as writer:
        update_summary(writer,reference,target)
        update_swinfo(writer)
        update_failures(writer)
        update_details(writer)

if __name__ == '__main__':
    generate_report(args.reference, args.target)