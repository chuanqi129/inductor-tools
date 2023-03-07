"""log_parser.py
Generate report or data compare report from specified inductor logs.
Usage:
  python log_parser.py --reference WW48.2 --target WW48.4 -m multiple
"""


import argparse
from styleframe import StyleFrame, Styler, utils
import pandas as pd
import os

parser = argparse.ArgumentParser(description="Generate report from two specified inductor logs")
parser.add_argument('-t','--target',type=str,help='target log file')
parser.add_argument('-r','--reference',type=str,help='reference log file')
parser.add_argument('-m','--mode',type=str,help='multiple or single mode')
args=parser.parse_args()

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
        if args.mode == "multiple" or args.mode is None:
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
        if args.mode == "single" or args.mode is None:
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
        if args.mode == "multiple" or args.mode is None:
            target_mt_pr_data=pd.read_csv(target_mt+'/passrate.csv',index_col=0)
            target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0)
            summary.iloc[0:1,4:7]=target_mt_pr_data.iloc[0:2,1:7]
            summary.iloc[1:2,4:7]=target_mt_gm_data.iloc[0:2,1:7] 
            summary.iloc[0:1,2]=target
            summary.iloc[1:2,2]=target
        if args.mode == "single" or args.mode is None:
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

def update_swinfo(excel):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata','dynamo_benchmarks'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' ',' '],'Master/Main commit':[' ', ' ', ' ', ' ',' ',' ',' ']}
    swinfo=pd.DataFrame(data)

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)   

def update_failures(excel,target_thread):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=target_thread+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
        acc_path=target_thread+'/inductor_'+suite+'_float32_inference_cpu_accuracy.csv'

        perf_data=pd.read_csv(perf_path)
        acc_data=pd.read_csv(acc_path)
        
        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
        tmp.append(acc_data)

        perf_data=perf_data.loc[perf_data['speedup'] ==0,:]
        tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['name','accuracy','speedup']]

    failures['accuracy'].replace(["fail_to_run","fail_accuracy","0.0000"],["X","X","X"],inplace=True)
    failures['speedup'].replace([0],["X"],inplace=True)
    
    failures=failures.rename(columns={'name':'name','accuracy':'accuracy','speedup':'perf'})
    failures=failures.groupby('name').sum().reset_index()
    failures['perf'].replace([0],["v"],inplace=True)
    failures['accuracy'].replace([0],["v"],inplace=True)

    sf = StyleFrame({'name': list(failures['name']),
                 'accuracy': list(failures['accuracy']),
                 'perf': list(failures['perf']),
                 'reason':[''] * len(list(failures['name'])),
                 'repro cmd':[''] * len(list(failures['name']))})
    
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['accuracy'] == "X"],
                            cols_to_style='accuracy',
                            styler_obj=failed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['perf'] == "X"],
                            cols_to_style='perf',
                            styler_obj=failed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['accuracy'] == "v"],
                            cols_to_style='accuracy',
                            styler_obj=passed_style,
                            overwrite_default_style=False)
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['perf'] == "v"],
                            cols_to_style='perf',
                            styler_obj=passed_style,
                            overwrite_default_style=False)    
    sf.set_column_width(1, 30)
    sf.set_column_width(2, 15)
    sf.set_column_width(3, 15)    
    sf.set_column_width(4, 100) 
    sf.set_column_width(5, 150)              

    sf.to_excel(sheet_name='Failures in '+target_thread.split('_cf')[0].split('/')[2].strip(),excel_writer=excel,index=False)

def process_suite(suite,thread):
    target_file_path=getfolder(args.target,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
    target_ori_data=pd.read_csv(target_file_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency']]
    target_data=target_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)

    if args.reference is not None:
        reference_file_path=getfolder(args.reference,thread)+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
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
        data_new['eager_new'] = data_new['speed_up_new'] * data_new['inductor_new']        
        data_old=input[['batch_size_y','speedup_y','abs_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old'})    
        data_old['inductor_old']=data_old['inductor_old'].astype(float).div(1000)
        data_old['eager_old'] = data_old['speed_up_old'] * data_old['inductor_old']
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
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] < 0.9],styler_obj=regression_style)
        data.set_row_height(rows=data.row_indexes, height=15)    
    else:
        data_new=input[['name','batch_size','speedup','abs_latency']].rename(columns={'name':'name','batch_size':'batch_size','speedup':'speedup',"abs_latency":'inductor'})
        data_new['inductor']=data_new['inductor'].astype(float).div(1000)
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

    if args.mode == "multiple" or args.mode is None:
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
    if args.mode == "single" or args.mode is None:
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

def update_md_file(target):
    if args.mode is None:
        for thread in ['multi_threads_cf_logs','single_thread_cf_logs']:
            result=open(thread.split('_')[0]+'_'+target+'.md','a+')
            folder = getfolder(target,thread)
            title=folder+'/gh_title.txt'
            summary=folder+'/gh_executive_summary.txt'
            inference=folder+'/gh_inference.txt'

            for file in title,summary,inference:
                for line in open(file,'r'):
                    result.writelines(line)

def generate_report(excel,reference,target):
    update_summary(excel,reference,target)
    update_swinfo(excel)
    if args.mode == 'multiple' or args.mode is None:
        update_failures(excel,target_mt)
    if args.mode =='single' or args.mode is None:
        update_failures(excel,target_st)
    update_details(excel)
    update_md_file(target)

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
    if args.mode == "multiple" or args.mode is None:  
        # Single-Socket Multi-threads
        wmt=wb['Single-Socket Multi-threads']
        wmt.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=torchbench_index+3,end_row=hf_index+2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=hf_index+3,end_row=timm_index+2,start_column=1,end_column=1)
        wmt.merge_cells(start_row=timm_index+3,end_row=end_index+3,start_column=1,end_column=1)
        wmt.merge_cells(start_row=1,end_row=1,start_column=3,end_column=6)
        wmt.merge_cells(start_row=1,end_row=1,start_column=7,end_column=10)
        wmt.merge_cells(start_row=1,end_row=1,start_column=11,end_column=13)
    if args.mode == "single" or args.mode is None:         
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
    excel = StyleFrame.ExcelWriter('Inductor Dashboard Regression Check '+args.target+'.xlsx')
    generate_report(excel,args.reference, args.target)
    excel_postprocess(excel)