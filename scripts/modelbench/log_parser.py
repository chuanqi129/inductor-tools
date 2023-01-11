"""log_parser.py
Generate report or data compare report from specified inductor logs.
Usage:
  python log_parser.py --reference WW48.2 --target WW48.4
  python log_parser.py --target WW48.4

"""


import argparse
from styleframe import StyleFrame, Styler, utils
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

if args.reference is not None:
    reference_mt=getfolder(args.reference,'multi_threads_cf_logs')
    reference_st=getfolder(args.reference,'single_thread_cf_logs')
target_mt=getfolder(args.target,'multi_threads_cf_logs')
target_st=getfolder(args.target,'single_thread_cf_logs')

target_style = Styler(bg_color='#DCE6F1', font_color=utils.colors.black)
red_style = Styler(bg_color='#FF0000', font_color=utils.colors.black)

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
    summary=pd.DataFrame(data)
    if reference is not None:
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

    sf = StyleFrame(summary)
    sf.apply_style_by_indexes(sf.index[[1,3,5,7]], styler_obj=target_style) 
    for i in range(1,8):
        sf.set_column_width(i, 15)

    sf.to_excel(sheet_name='Summary',excel_writer=excel)

def update_swinfo(excel):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata'], 'Nightly commit':[' ', '/', ' ', ' ',' ',' '],'Master/Main commit':[' ', ' ', ' ', ' ',' ',' ']}
    swinfo=pd.DataFrame(data)

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 18)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)   

def update_failures(excel):
    tmp=[]
    for suite in 'torchbench','huggingface','timm_models':
        perf_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_performance.csv'
        acc_path=target_mt+'/inductor_'+suite+'_float32_inference_cpu_accuracy.csv'

        perf_data=pd.read_csv(perf_path)
        acc_data=pd.read_csv(acc_path)
        
        acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
        tmp.append(acc_data)

        perf_data=perf_data.loc[perf_data['speedup'] ==0,:]
        tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['name','accuracy','speedup']]

    failures['accuracy'].replace(["fail_to_run","fail_accuracy","0"],["X","X","X"],inplace=True)
    failures['speedup'].replace([0],["X"],inplace=True)
    
    failures=failures.rename(columns={'name':'name','accuracy':'accuracy','speedup':'perf'})
    
    sf = StyleFrame({'name': list(failures['name']),
                 'accuracy': list(failures['accuracy']),
                 'perf': list(failures['perf']),
                 'reason':[''] * len(list(failures['name'])),
                 'repro cmd':[''] * len(list(failures['name']))})

    sf.set_column_width(1, 30)
    sf.set_column_width(2, 15)
    sf.set_column_width(3, 15)    
    sf.set_column_width(4, 15) 
    sf.set_column_width(5, 15)              

    sf.to_excel(sheet_name='Failures',excel_writer=excel,index=False)


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
        data.set_column_width(4, 15)
        data.set_column_width(5, 15)
        data.set_column_width(6, 18)
        data.set_column_width(7, 18) 
        data.set_column_width(8, 18) 
        data.set_column_width(9, 15)
        data.set_column_width(10, 10) 
        data.set_column_width(11, 22) 
        data.set_column_width(12, 25) 
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size_new'] == 0], styler_obj=red_style) 
        #data.add_color_scale_conditional_formatting(start_type=None, start_value=0, start_color='0xff00ff00', end_type=None, end_value=None, end_color='0xFFFF0000', mid_type=None, mid_value=None, mid_color='0xFFFFFF00')
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
        data.set_column_width(4, 15)
        data.set_column_width(5, 15)
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size'] == 0], styler_obj=red_style) 
        #data.add_color_scale_conditional_formatting(start_type=None, start_value=0, start_color='0xff00ff00', end_type=None, end_value=None, end_color='0xFFFF0000', mid_type=None, mid_value=None, mid_color='0xFFFFFF00')
        data.set_row_height(rows=data.row_indexes, height=15)        
    return data

def update_details(writer):
    h = {"A": '', "B": args.target, "C": '', "D": '',"E": '', "F": args.reference, "G": '', "H": '',"I": '',"J": 'Result Comp',"K": '',"L": ''}
    if args.reference is None:
        h = {"A": '', "B": args.target, "C": '', "D": '',"E": ''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_row_height(rows=[1], height=15)

    # mt
    head.to_excel(excel_writer=writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
    mt=process_thread('multi_threads_cf_logs')
    process(mt).to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=0) 
          
    # st
    head.to_excel(excel_writer=writer, sheet_name='Single-Socket Single-thread', index=False,startrow=0,header=False) 
    st=process_thread('single_thread_cf_logs')
    process(st).to_excel(sheet_name='Single-Socket Single-thread',excel_writer=writer,index=False,startrow=1,startcol=0)

    writer.save()    

def generate_report(excel,reference,target):
    update_summary(excel,reference,target)
    update_swinfo(excel)
    update_failures(excel)
    update_details(excel)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter('Inductor Dashboard Regression Check '+args.target+'.xlsx')
    generate_report(excel,args.reference, args.target)