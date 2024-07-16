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
import json

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
parser.add_argument('--shape',type=str,default='static',help='Shape: static or dynamic')
parser.add_argument('--wrapper',type=str,default='default',help='Wrapper: default or cpp')
parser.add_argument('--torch_repo',type=str,default='https://github.com/pytorch/pytorch.git',help='pytorch repo')
parser.add_argument('--torch_branch',type=str,default='main',help='pytorch branch')
parser.add_argument('--backend',type=str, help='pytorch dynamo backend')
parser.add_argument('--ref_backend',type=str, default='inductor', help='reference backend for comparsion')
parser.add_argument('--threshold',type=float, default=0.1,help='threshold for checking performance regression and improvement')
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

new_performance_regression=pd.DataFrame()
new_performance_regression_model_list=pd.DataFrame()
new_failures=pd.DataFrame()
new_failures_model_list=pd.DataFrame()
new_performance_improvement=pd.DataFrame()
new_performance_improvement_model_list=pd.DataFrame()
new_fixed_failures=pd.DataFrame()
new_fixed_failures_model_list=pd.DataFrame()
target_thread_failures=pd.DataFrame()

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

start_commit = "None"
end_commit = "None"

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

def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)

def update_passrate_csv(df, target_path, backend):
    new_df = df.copy()
    for suite_name in suite_list:
        passrate_str = new_df.loc[backend][suite_name]
        passed_num = int(passrate_str.split(', ')[1].split('/')[0])
        perf_path = '{0}/{1}_{2}_{3}_{4}_cpu_performance.csv'.format(target_path, backend, suite_name, args.precision, args.infer_or_train)
        acc_path = '{0}/{1}_{2}_{3}_{4}_cpu_accuracy.csv'.format(target_path, backend, suite_name, args.precision, args.infer_or_train)
        perf_df = pd.read_csv(perf_path)
        acc_df = pd.read_csv(acc_path)
        acc_df = acc_df.drop(acc_df[(acc_df['accuracy'] == 'model_fail_to_load') | (acc_df['accuracy'] == 'eager_fail_to_run')].index)
        name_union_df = pd.merge(acc_df['name'], perf_df['name'], how='left')
        perc = int(percentage(passed_num, len(name_union_df), decimals=0))
        passrate_str_new = '{0}%, {1}/{2}'.format(perc, passed_num, len(name_union_df))
        new_df.loc[backend][suite_name] = passrate_str_new
    new_df.to_csv(target_path + '/passrate_new.csv')

def update_passrate(reference):
    if reference is not None:
        if args.mode == "multiple" or args.mode == 'all':
            reference_mt_pr_data = pd.read_csv(reference_mt+'/passrate.csv',index_col=0)
            update_passrate_csv(reference_mt_pr_data, reference_mt, args.ref_backend)
            target_mt_pr_data = pd.read_csv(target_mt+'/passrate.csv',index_col=0)
            update_passrate_csv(target_mt_pr_data, target_mt, args.backend)
        if args.mode == "single" or args.mode == 'all':
            reference_st_pr_data = pd.read_csv(reference_st+'/passrate.csv',index_col=0)
            update_passrate_csv(reference_st_pr_data, reference_st, args.ref_backend)
            target_st_pr_data = pd.read_csv(target_st+'/passrate.csv',index_col=0)
            update_passrate_csv(target_st_pr_data, target_st, args.backend)
    else:
        if args.mode == "multiple" or args.mode == 'all':
            target_mt_pr_data=pd.read_csv(target_mt+'/passrate.csv',index_col=0)
            update_passrate_csv(target_mt_pr_data, target_mt, args.backend)
        if args.mode == "single" or args.mode == 'all':
            target_st_pr_data=pd.read_csv(target_st+'/passrate.csv',index_col=0)
            update_passrate_csv(target_st_pr_data, target_st, args.backend)

def update_summary(excel, reference, target, passrate_file, sheet_name):
    if args.suite == 'all':
        data = {
            'Test Scenario':['Single Socket Multi-Threads', ' ', ' ', ' ','Single Core Single-Thread',' ',' ',' '], 
            'Comp Item':['Pass Rate', ' ', 'Geomean Speedup', ' ','Pass Rate',' ','Geomean Speedup',' '],
            'Date':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor','inductor','inductor','inductor'],
            'torchbench':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
            'huggingface':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
            'timm_models ':[' ', ' ', ' ', ' ',' ',' ',' ',' ']
        }
        data_target = {
            'Test Scenario':['Single Socket Multi-Threads', ' ','Single Core Single-Thread',' '], 
            'Comp Item':['Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup'],
            'Date':[' ', ' ', ' ', ' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor'],
            'torchbench':[' ', ' ', ' ', ' '],
            'huggingface':[' ', ' ', ' ', ' '],
            'timm_models ':[' ', ' ', ' ', ' ']
        }
    else:
        data = {
            'Test Scenario':['Single Socket Multi-Threads', ' ', ' ', ' ','Single Core Single-Thread',' ',' ',' '], 
            'Comp Item':['Pass Rate', ' ', 'Geomean Speedup', ' ','Pass Rate',' ','Geomean Speedup',' '],
            'Date':[' ', ' ', ' ', ' ',' ',' ',' ',' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor','inductor','inductor','inductor'],
            args.suite:[' ', ' ', ' ', ' ',' ',' ',' ',' ']
        }
        data_target = {
            'Test Scenario':['Single Socket Multi-Threads', ' ','Single Core Single-Thread',' '], 
            'Comp Item':['Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup'],
            'Date':[' ', ' ', ' ', ' '],
            'Compiler':['inductor', 'inductor', 'inductor', 'inductor'],
            args.suite:[' ', ' ', ' ', ' ']
        }

    if reference is not None:
        summary=pd.DataFrame(data)
        if args.mode == "multiple" or args.mode == 'all':
            reference_mt_pr_data=pd.read_csv(reference_mt+'/'+passrate_file,index_col=0)
            reference_mt_gm_data=pd.read_csv(reference_mt+'/geomean.csv',index_col=0)
            summary.iloc[0:1,4:7]=reference_mt_pr_data.iloc[0:1,1:7]
            summary.iloc[2:3,4:7]=reference_mt_gm_data.iloc[0:2,1:7] 
            summary.iloc[0:1,2]=reference
            summary.iloc[2:3,2]=reference 
            target_mt_pr_data=pd.read_csv(target_mt+'/'+passrate_file,index_col=0)
            target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0) 
            summary.iloc[1:2,4:7]=target_mt_pr_data.iloc[0:1,1:7]
            summary.iloc[3:4,4:7]=target_mt_gm_data.iloc[0:2,1:7]
            summary.iloc[1:2,2]=target
            summary.iloc[3:4,2]=target
        if args.mode == "single" or args.mode == 'all':
            reference_st_pr_data=pd.read_csv(reference_st+'/'+passrate_file,index_col=0)
            reference_st_gm_data=pd.read_csv(reference_st+'/geomean.csv',index_col=0)
            summary.iloc[4:5,4:7]=reference_st_pr_data.iloc[0:1,1:7]
            summary.iloc[6:7,4:7]=reference_st_gm_data.iloc[0:2,1:7]
            summary.iloc[4:5,2]=reference
            summary.iloc[6:7,2]=reference
            target_st_pr_data=pd.read_csv(target_st+'/'+passrate_file,index_col=0)
            target_st_gm_data=pd.read_csv(target_st+'/geomean.csv',index_col=0)
            summary.iloc[5:6,4:7]=target_st_pr_data.iloc[0:1,1:7]
            summary.iloc[7:8,4:7]=target_st_gm_data.iloc[0:2,1:7]
            summary.iloc[5:6,2]=target
            summary.iloc[7:8,2]=target
        sf = StyleFrame(summary)
        sf.apply_style_by_indexes(sf.index[[1,3,5,7]], styler_obj=target_style) 
    else:
        summary=pd.DataFrame(data_target)
        if args.mode == "multiple" or args.mode == 'all':
            target_mt_pr_data=pd.read_csv(target_mt+'/'+passrate_file,index_col=0)
            target_mt_gm_data=pd.read_csv(target_mt+'/geomean.csv',index_col=0)
            if args.suite == 'all':
                summary.iloc[0:1,4:7]=target_mt_pr_data.iloc[0:1,1:7]
                summary.iloc[1:2,4:7]=target_mt_gm_data.iloc[0:2,1:7]
            else:
                summary.iloc[0,4]=target_mt_pr_data.iloc[0,1]
                summary.iloc[1,4]=target_mt_gm_data.iloc[0,1]
            summary.iloc[0:1,2]=target
            summary.iloc[1:2,2]=target
        if args.mode == "single" or args.mode == 'all':
            target_st_pr_data=pd.read_csv(target_st+'/'+passrate_file,index_col=0)
            target_st_gm_data=pd.read_csv(target_st+'/geomean.csv',index_col=0)
            if args.suite == 'all':
                summary.iloc[2:3,4:7]=target_st_pr_data.iloc[0:1,1:7]
                summary.iloc[3:4,4:7]=target_st_gm_data.iloc[0:2,1:7]
            else:
                summary.iloc[2,4]=target_st_pr_data.iloc[0,1]
                summary.iloc[3,4]=target_st_gm_data.iloc[0,1]
            summary.iloc[2:3,2]=target
            summary.iloc[3:4,2]=target
        sf = StyleFrame(summary)
    for i in range(1, len(data)+1):
        sf.set_column_width(i, 18)
    sf.to_excel(sheet_name=sheet_name,excel_writer=excel)

def update_swinfo(excel):
    if not (os.path.exists(args.target+'/inductor_log/version.csv')):
        print("target version.csv not found")
        return
    refer_read_flag = True
    global start_commit
    global end_commit
    try:
        swinfo_df = pd.read_csv(args.target+'/inductor_log/version.csv')
        swinfo_df = swinfo_df.rename(columns={'branch':'target_branch','commit':'target_commit'})
        start_commit = swinfo_df.loc[swinfo_df['name'] == 'torch', 'target_commit'].values[0]
        if args.reference is not None:
            refer_swinfo_df = pd.read_csv(args.reference+'/inductor_log/version.csv')
            refer_swinfo_df = refer_swinfo_df.rename(columns={'branch':'refer_branch','commit':'refer_commit'})
            swinfo_df = pd.merge(swinfo_df, refer_swinfo_df)
            end_commit = swinfo_df.loc[swinfo_df['name'] == 'torch', 'refer_commit'].values[0]
    except :
        print("referece version.csv not found")
        swinfo_df = pd.read_csv(args.target+'/inductor_log/version.csv')
        swinfo_df = swinfo_df.rename(columns={'branch':'target_branch','commit':'target_commit'})
        start_commit = swinfo_df.loc[swinfo_df['name'] == 'torch', 'target_commit'].values[0]
        refer_read_flag = False

    sf = StyleFrame(swinfo_df)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)
    if refer_read_flag and (args.reference is not None):
        sf.set_column_width(4, 20)
        sf.set_column_width(5, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)

# only accuracy failures
def parse_acc_failure(file,failed_model):
    result = []
    found = False
    skip = False
    if failed_model in known_failures.keys():
        result.append(failed_model+", "+known_failures[failed_model])
    else:
        # if not ignore errors, aot_inductor log will throw
        # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 3122: invalid start byte
        with open(file, 'r', errors='ignore') as reader:
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
                if found ==  True and ("Error: " in line or "[ERROR]" in line or "TIMEOUT" in line or "FAIL" in line or "fail" in line):
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
        with open(file, 'r', errors='ignore') as reader:
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

def failures_reason_parse(model, acc_or_perf, mode):
    raw_log_pre="multi_threads_model_bench_log" if "multi" in mode else "single_thread_model_bench_log"
    raw_log=getfolder(args.target,raw_log_pre)
    content=parse_acc_failure(raw_log,model) if acc_or_perf =="accuracy" else parse_failure(raw_log,model)
    ct_dict=str_to_dict(content)
    try:
        line = ct_dict[model]
    except KeyError:
        line=''
        pass
    return line

def get_failures(target_path, thread_mode, backend_pattern):
    all_model_df = pd.DataFrame()
    failure_msg_list = [
        'fail_to_run',
        'infra_error',
        'fail_accuracy',
        'eager_fail_to_run',
        'model_fail_to_load',
        'eager_two_runs_differ',
        'timeout',
        '0.0000']
    for suite_name in suite_list:
        perf_path = '{0}/{1}_{2}_{3}_{4}_cpu_performance.csv'.format(target_path, backend_pattern, suite_name, args.precision, args.infer_or_train)
        acc_path = '{0}/{1}_{2}_{3}_{4}_cpu_accuracy.csv'.format(target_path, backend_pattern, suite_name, args.precision, args.infer_or_train)

        perf_data = pd.read_csv(perf_path, usecols=["name", "batch_size", "speedup"]).rename(columns={'batch_size': 'perf_bs'})
        acc_data = pd.read_csv(acc_path, usecols=["name", "batch_size", "accuracy"]).rename(columns={'batch_size': 'acc_bs'})
        all_data = pd.merge(perf_data, acc_data, how='outer')
        all_data.insert(loc=0, column='suite', value=suite_name)
        all_model_df = pd.concat([all_model_df, all_data])

    all_model_df.to_csv("all_model_list.csv", index=False)
    failures = all_model_df.loc[(all_model_df['accuracy'].isin(failure_msg_list))
                                   | (all_model_df['acc_bs'] == 0)
                                   | (all_model_df['speedup'] == 'infra_error') 
                                   | (all_model_df['speedup'] == 0) 
                                   | (all_model_df['perf_bs'] == 0) 
                                   | (all_model_df['acc_bs'].isna()) 
                                   | (all_model_df['perf_bs'].isna())]
    
    # There is no failure in accuracy and performance, just return
    if (len(failures) == 0):
        return failures

    failures = failures.rename(columns={'speedup': 'perf'})
    failures = failures[['suite', 'name', 'accuracy', 'perf']]
    failures.replace(failure_msg_list, [0]*len(failure_msg_list), inplace=True)
    failures['accuracy'].replace('pass', 1, inplace=True)
    failures.loc[(failures['perf'] > 0), ['perf']] = 1
    failures.fillna(2, inplace=True)
    failures.replace([0, 1, 2],["X", "√", "N/A"],inplace=True)
    
    # fill failure reasons
    failures_dict = {}
    for key in failures['name']:
        if failures.loc[(failures['name'] == key), ['accuracy']].values[0] == "X":
            failures_dict[key] = 'accuracy'
        else:
            failures_dict[key] = 'perf'
    reason_content=[]
    for model in failures_dict.keys():
        reason_content.append(failures_reason_parse(model, failures_dict[model], target_path))
    failures['reason(reference only)'] = reason_content
    failures['thread'] = thread_mode
    col_order = ['suite', 'name', 'thread', 'accuracy', 'perf', 'reason(reference only)']
    failures = failures[col_order]
    return failures

def get_fail_model_list(failures, thread, kind):
    accuracy_model_list = failures.loc[(failures['accuracy'] == "X")]
    accuracy_model_list['accuracy'].replace(["X"], ['accuracy'], inplace=True)
    accuracy_model_list = accuracy_model_list.rename(columns={'suite':'suite','name':'name','accuracy':'scenario'})
    accuracy_model_list = accuracy_model_list[['suite','name','scenario']]

    perf_model_list = failures.loc[(failures['perf'] == "X")]
    perf_model_list['perf'].replace(["X"], ['performance'], inplace=True)
    perf_model_list = perf_model_list.rename(columns={'suite':'suite','name':'name','perf':'scenario'})
    perf_model_list = perf_model_list[['suite','name','scenario']]

    model_list = pd.concat([accuracy_model_list, perf_model_list])
    model_list['thread'] = thread
    model_list['kind'] = kind
    model_list['precision'] = args.precision
    return model_list

def get_perf_model_list(regression, thread, kind):
    model_list = regression[['suite','name']]
    model_list['scenario'] = 'performance'
    model_list['thread'] = thread
    model_list['kind'] = kind
    model_list['precision'] = args.precision
    return model_list

def update_failures(excel, target_thread, refer_thread, thread_mode):
    global new_failures
    global new_fixed_failures
    global new_failures_model_list
    global new_fixed_failures_model_list
    global target_thread_failures
    target_thread_failures = get_failures(target_thread, thread_mode, backend_pattern=args.backend)
    # new failures compare with reference logs
    if args.reference is not None:
        refer_thread_failures = get_failures(refer_thread, thread_mode, backend_pattern=args.ref_backend)
        # New Failures
        failure_regression_compare = datacompy.Compare(target_thread_failures, refer_thread_failures, join_columns='name')
        failure_regression = failure_regression_compare.df1_unq_rows.copy()
        new_failures = pd.concat([new_failures,failure_regression])
        model_list = get_fail_model_list(failure_regression, thread_mode, 'crash')
        if not model_list.empty:
            new_failures_model_list = pd.concat([new_failures_model_list, model_list])

        # Fixed Failures
        fixed_failures_compare = datacompy.Compare(refer_thread_failures, target_thread_failures, join_columns='name')
        fixed_failures = fixed_failures_compare.df1_unq_rows.copy()
        new_fixed_failures = pd.concat([new_fixed_failures,fixed_failures])
        model_list = get_fail_model_list(fixed_failures, thread_mode, 'fixed')
        if not model_list.empty:
            new_fixed_failures_model_list = pd.concat([new_fixed_failures_model_list, model_list])

    # There is no failure in target, just return
    if (len(target_thread_failures) == 0):
        return
    target_thread_failures['thread'] = thread_mode
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

def process_suite(suite, thread):
    target_file_path = '{0}/{1}_{2}_{3}_{4}_cpu_performance.csv'.format(getfolder(args.target, thread), args.backend, suite, args.precision, args.infer_or_train)
    target_ori_data=pd.read_csv(target_file_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency','compilation_latency']]
    target_data=target_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)

    if args.reference is not None:
        reference_file_path = '{0}/{1}_{2}_{3}_{4}_cpu_performance.csv'.format(getfolder(args.reference, thread), args.ref_backend, suite, args.precision, args.infer_or_train)
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
 
def process(input, thread):
    if args.reference is not None:
        data_new=input[['suite','name','batch_size_x','speedup_x','abs_latency_x','compilation_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new',"compilation_latency_x":'compilation_latency_new'})
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
            'suite': list(data_new['suite']),
            'name': list(data_new['name']),
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
        data.apply_style_by_indexes(indexes_to_style=data[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < (1 - args.threshold))],styler_obj=regression_style)
        global new_performance_regression
        global new_performance_regression_model_list
        regression = data.loc[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < (1 - args.threshold))]
        regression = regression.copy()
        regression.insert(2, 'thread', thread)
        new_performance_regression = pd.concat([new_performance_regression,regression])
        model_list = get_perf_model_list(regression, thread, 'drop')
        if not model_list.empty:
            new_performance_regression_model_list = pd.concat([new_performance_regression_model_list, model_list])
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] > (1 + args.threshold)],styler_obj=improve_style)
        data.set_row_height(rows=data.row_indexes, height=15)

        global new_performance_improvement
        global new_performance_improvement_model_list
        improvement = data.loc[(data['Inductor Ratio(old/new)'] > (1 + args.threshold))]
        improvement = improvement.copy()
        improvement.insert(2, 'thread', thread)
        new_performance_improvement = pd.concat([new_performance_improvement, improvement])
        model_list = get_perf_model_list(improvement, thread, 'improve')
        if not model_list.empty:
            new_performance_improvement_model_list = pd.concat([new_performance_improvement_model_list, model_list])
    else:
        data_new=input[['suite','name','batch_size','speedup','abs_latency','compilation_latency']].rename(columns={'name':'name','batch_size':'batch_size','speedup':'speedup',"abs_latency":'inductor',"compilation_latency":'compilation_latency'})
        data_new['inductor']=data_new['inductor'].astype(float).div(1000)
        data_new['speedup']=data_new['speedup'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager'] = data_new['speedup'] * data_new['inductor']        
        data = StyleFrame({
            'suite': list(data_new['suite']),
            'name': list(data_new['name']),
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
    h = {"A": 'Suite', "B": 'Model', "C": args.target, "D": '', "E": '',"F": '', "G": '',"H": args.reference, "I": '', "J": '',"K": '',"L":'',"M": 'Result Comp',"N": '',"O": '',"P":''}
    if args.reference is None:
        h = {"A": 'Suite', "B": 'Model', "C": args.target, "D": '', "E": '',"F": '', "G": ''}
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

def update_cppwrapper_gm(excel,reference,target):
    # cppwrapper vs pythonwrapper geomean speedup table
    cppwrapper_gm = {
        'Test Scenario':['Single Socket Multi-Threads', 'Test Scenario','Single Core Single-Thread'], 
        'Comp Item':['Geomean Speedup','Comp Item','Geomean Speedup'],
        'Date':[' ', ' Date', ' '],
        'Compiler':['inductor', 'Compiler', 'inductor'],
        f'small(t<={args.mt_interval_start}s)':[' ', f'small(t<={args.st_interval_start}s)', ' '],
        f'medium({args.mt_interval_start}s<t<={args.mt_interval_end}s)':[' ', f'medium({args.st_interval_start}s<t<={args.st_interval_end}s)', ' '],
        f'large(t>{args.mt_interval_end}s)':[' ', f'large(t>{args.st_interval_end}s)', ' ']
    }
    if reference is not None:
        cppwrapper_summary=pd.DataFrame(cppwrapper_gm)
        if args.mode == "multiple" or args.mode == 'all':
            cppwrapper_summary.iloc[0:1,4:5]=multi_threads_gm['small']
            cppwrapper_summary.iloc[0:1,5:6]=multi_threads_gm['medium']
            cppwrapper_summary.iloc[0:1,6:7]=multi_threads_gm['large']
            cppwrapper_summary.iloc[0:1,2]=target
            cppwrapper_summary.iloc[0:1,2]=target          
        if args.mode == "single" or args.mode == 'all':
            cppwrapper_summary.iloc[2:3,4:5]=single_thread_gm['small']
            cppwrapper_summary.iloc[2:3,5:6]=single_thread_gm['medium']
            cppwrapper_summary.iloc[2:3,6:7]=single_thread_gm['large']
            cppwrapper_summary.iloc[2:3,2]=target
            cppwrapper_summary.iloc[2:3,2]=target                    
        cppwrapper_sf = StyleFrame(cppwrapper_summary)

    for i in range(1,8):
        cppwrapper_sf.set_column_width(i, 30) if i==6 else cppwrapper_sf.set_column_width(i, 18)
    cppwrapper_sf.to_excel(sheet_name='Cppwrapper_GM',excel_writer=excel)

def update_issue_commits(precision):
    from github import Github
    # generate md files
    icx_hw_info=f'''
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
'''
    spr_hw_info=f'''
|  Item  | Value  |
|  ----  | ----  |
| Manufacturer  | Amazon EC2 |
| Product Name  | c7i.metal-24xl |
| CPU Model  | Intel(R) Xeon(R) Platinum 8488C CPU @ 2.40GHz |
| Installed Memory  | 192GB (8x24GB DDR5 4800 MT/s [4800 MT/s]) |
| OS  | Ubuntu 22.04.3 LTS |
| Kernel  | 6.2.0-1017-aws |
| Microcode  | 0x2b0004d0 |
| GCC  | gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 |
| GLIBC  | ldd (Ubuntu GLIBC 2.35-0ubuntu3.4) 2.35 |
| Binutils  | GNU ld (GNU Binutils for Ubuntu) 2.38 |
| Python  | Python 3.8.18 |
| OpenSSL  | OpenSSL 3.2.0 23 Nov 2023 (Library: OpenSSL 3.2.0 23 Nov 2023) |
'''
    hw_info = ""
    if precision == "float32":
        hw_info = icx_hw_info
    else:
        hw_info = spr_hw_info
    print(hw_info)
    
    # Software information
    swinfo_df = pd.read_csv(args.target+'/inductor_log/version.csv')
    swinfo_df.set_index('name', inplace=True)
    torch_branch = swinfo_df.at['torch', 'branch']
    torch_commit = swinfo_df.at['torch', 'commit']
    torchbench_branch = swinfo_df.at['torchbench', 'branch']
    torchbench_commit = swinfo_df.at['torchbench', 'commit']
    torchvision_branch = swinfo_df.at['torchvision', 'branch']
    torchvision_commit = swinfo_df.at['torchvision', 'commit']
    torchtext_branch = swinfo_df.at['torchtext', 'branch']
    torchtext_commit = swinfo_df.at['torchtext', 'commit']
    torchaudio_branch = swinfo_df.at['torchaudio', 'branch']
    torchaudio_commit = swinfo_df.at['torchaudio', 'commit']
    torchdata_branch = swinfo_df.at['torchdata', 'branch']
    torchdata_commit = swinfo_df.at['torchdata', 'commit']
    dynamo_benchmarks_branch = swinfo_df.at['dynamo_benchmarks', 'branch']
    dynamo_benchmarks_commit = swinfo_df.at['dynamo_benchmarks', 'commit']
    sw_info = f'''
SW information:

SW	| Branch | Commit
-- | -- | --
Pytorch|[{torch_branch}](https://github.com/pytorch/pytorch/tree/{torch_branch})|[{torch_commit}](https://github.com/pytorch/pytorch/commit/{torch_commit})
Torchbench|[{torchbench_branch}](https://github.com/pytorch/benchmark/tree/{torchbench_branch})|[{torchbench_commit}](https://github.com/pytorch/benchmark/commit/{torchbench_commit})
torchaudio|[{torchaudio_branch}](https://github.com/pytorch/audio/tree/{torchaudio_branch})|[{torchaudio_commit}](https://github.com/pytorch/audio/commit/{torchaudio_commit})
torchtext|[{torchtext_branch}](https://github.com/pytorch/text/tree/{torchtext_branch})| [{torchtext_commit}](https://github.com/pytorch/text/commit/{torchtext_commit})
torchvision|[{torchvision_branch}](https://github.com/pytorch/vision/tree/{torchvision_branch})|[{torchvision_commit}](https://github.com/pytorch/vision/commit/{torchvision_commit})
torchdata|[{torchdata_branch}](https://github.com/pytorch/data/tree/{torchdata_branch})|[{torchdata_commit}](https://github.com/pytorch/data/commit/{torchdata_commit})
dynamo_benchmarks|[{dynamo_benchmarks_branch}](https://github.com/pytorch/pytorch/tree/{dynamo_benchmarks_branch})|[{dynamo_benchmarks_commit}](https://github.com/pytorch/pytorch/commit/{dynamo_benchmarks_commit})


HW information

{hw_info}
'''
    mt_addtional= sw_info +'''
Test command

```bash
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export TORCHINDUCTOR_FREEZING=1
CORES=$(lscpu | grep Core | awk '{print $4}')
export OMP_NUM_THREADS=$CORES
''' + f'''
python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--node_id 0" --devices=cpu --dtypes=float32 --inference --compilers={args.backend} --extra-args="--timeout 9000" 

```
'''
    st_addtional = sw_info + '''
Test command

```bash
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export TORCHINDUCTOR_FREEZING=1
export OMP_NUM_THREADS=1
''' + f'''
python benchmarks/dynamo/runner.py --enable_cpu_launcher --cpu_launcher_args "--core_list 0 --ncores_per_instance 1" --devices=cpu --dtypes=float32 --inference --compilers={args.backend} --batch_size=1 --threads 1 --extra-args="--timeout 9000"

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
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<title> {args.backend} Regular Model Bench Report </title>
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
  <p><h3>{args.backend} Regular Model Bench Report </p></h3> '''

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
            if args.mode == 'all':
                content = pd.read_excel('{0}/inductor_log/{1} Dashboard Regression Check {0} {2}.xlsx'.format(args.target, args.backend, args.suite),sheet_name=[0,1,2,3,6])
            else:
                if target_thread_failures.empty:
                    content = pd.read_excel('{0}/inductor_log/{1} Dashboard Regression Check {0} {2}.xlsx'.format(args.target, args.backend, args.suite),sheet_name=[0,1,2,3])
                else:
                    content = pd.read_excel('{0}/inductor_log/{1} Dashboard Regression Check {0} {2}.xlsx'.format(args.target, args.backend, args.suite),sheet_name=[0,1,2,4])
            summary= pd.DataFrame(content[0]).to_html(classes="table",index = False)
            swinfo= pd.DataFrame(content[1]).to_html(classes="table",index = False)

            if args.mode == 'all':
                mt_failures= pd.DataFrame(content[2]).to_html(classes="table",index = False)
                st_failures= pd.DataFrame(content[3]).to_html(classes="table",index = False)
                failures_html = \
                    "<p>Multi-threads Failures</p>" + mt_failures + \
                    "<p>Single-thread Failures</p>" + st_failures
                summary_new = pd.DataFrame(content[6]).to_html(classes="table",index = False)
            elif args.mode == 'multiple':
                if target_thread_failures.empty:
                    failures_html = "<p>Multi-threads Failures</p>" + "None"
                    summary_new = pd.DataFrame(content[3]).to_html(classes="table",index = False)
                else:
                    mt_failures= pd.DataFrame(content[2]).to_html(classes="table",index = False)
                    failures_html = "<p>Multi-threads Failures</p>" + mt_failures
                    summary_new = pd.DataFrame(content[4]).to_html(classes="table",index = False)
            elif args.mode == 'single':
                if target_thread_failures.empty:
                    failures_html = "<p>Single-thread Failures</p>" + "None"
                    summary_new = pd.DataFrame(content[3]).to_html(classes="table",index = False)
                else:
                    st_failures= pd.DataFrame(content[2]).to_html(classes="table",index = False)
                    failures_html = "<p>Single-thread Failures</p>" + st_failures
                    summary_new = pd.DataFrame(content[4]).to_html(classes="table",index = False)
            perf_regression= new_performance_regression.to_html(classes="table",index = False)
            failures_regression= new_failures.to_html(classes="table",index = False)
            perf_improvement = new_performance_improvement.to_html(classes="table",index = False)
            fixed_failures = new_fixed_failures.to_html(classes="table",index = False)
            with open(args.target+'/inductor_log/inductor_model_bench.html',mode = "a") as f, \
                open(args.target+'/inductor_log/inductor_perf_regression.html',mode = "a") as perf_f, \
                open(args.target+'/inductor_log/inductor_failures.html',mode = "a") as failure_f, \
                open(args.target+'/inductor_log/inductor_perf_improvement.html',mode = "a") as perf_boost_f, \
                open(args.target+'/inductor_log/inductor_fixed_failures.html',mode = "a") as fixed_failure_f:
                f.write(html_head() + "<p>Summary</p>" + summary + \
                        "<p>Summary (exclude model_fail_to_load and eager_fail_to_run)</p>" + summary_new + \
                        "<p>SW info</p>" + swinfo + \
                        failures_html + \
                        "<h3><font color='#ff0000'>Regression</font></h3>" + \
                        "<p>new_perf_regression</p>" + perf_regression + \
                        "<p>new_failures</p>" + failures_regression + \
                        "<h3><font color='#00dd00'>Improvement</font></h3>" + \
                        "<p>new_perf_improvement</p>" + perf_improvement + \
                        "<p>new_fixed_failures</p>" + fixed_failures + \
                        f"<p>image: docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:{args.image_tag}</p>" + html_tail())
                perf_f.write(f"<p>new_perf_regression in {str((datetime.now() - timedelta(days=2)).date())}</p>" + \
                        perf_regression + "<p>SW info</p>" + swinfo + f"<p>image: docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:{args.image_tag}</p>")
                failure_f.write(f"<p>new_failures in {str((datetime.now() - timedelta(days=2)).date())}</p>" + \
                        failures_regression + "<p>SW info</p>" + swinfo + f"<p>image: docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:{args.image_tag}</p>")
                perf_boost_f.write(f"<p>new_perf_improvement in {str((datetime.now() - timedelta(days=2)).date())}</p>" + \
                        perf_improvement + "<p>SW info</p>" + swinfo + f"<p>image: docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:{args.image_tag}</p>")
                fixed_failure_f.write(f"<p>new_fixed_failures in {str((datetime.now() - timedelta(days=2)).date())}</p>" + \
                        fixed_failures + "<p>SW info</p>" + swinfo + f"<p>image: docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:{args.image_tag}</p>")
            f.close()
            perf_f.close()
            failure_f.close()
            perf_boost_f.close()
            fixed_failure_f.close()
        except:
            print("html_generate_failed")
            pass

def dump_common_info_json(json_file):
    common_info_dict = {}
    common_info_dict['shape'] = args.shape
    common_info_dict['wrapper'] = args.wrapper
    common_info_dict['torch_repo'] = args.torch_repo
    common_info_dict['torch_branch'] = args.torch_branch
    common_info_dict['start_commit'] = start_commit
    common_info_dict['end_commit'] = end_commit
    with open(json_file, 'w') as file:
        json.dump(common_info_dict, file, indent=4)

def generate_model_list():
    model_list = pd.concat([
        new_performance_regression_model_list,
        new_failures_model_list,
        new_performance_improvement_model_list,
        new_fixed_failures_model_list])
    model_list.to_csv("guilty_commit_search_model_list.csv", index=False)
    dump_common_info_json("guilty_commit_search_common_info.json")
    try:
        clean_model_list = pd.read_csv("guilty_commit_search_model_list.csv")
        clean_model_list.to_json('guilty_commit_search_model_list.json', indent=4, orient='records')
    except pd.errors.EmptyDataError:
        print('No new issue or improvement compared with reference')

def generate_report(excel, reference, target):
    update_passrate(reference)
    update_summary(excel, reference, target, 'passrate.csv', 'Summary')
    update_swinfo(excel)
    if args.mode == 'multiple' or args.mode == 'all':
        update_failures(excel, target_mt, reference_mt, 'multiple')
    if args.mode =='single' or args.mode == 'all':
        update_failures(excel, target_st, reference_st, 'single')
    update_details(excel)
    update_summary(excel, reference, target, 'passrate_new.csv', 'Summary New')
    generate_model_list()
    if args.cppwrapper_gm:
        update_cppwrapper_gm(excel,reference,target)

def excel_postprocess(file, sheet_name):
    wb=file.book
    # Summary
    ws=wb[sheet_name]
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
        wst.merge_cells(start_row=1,end_row=2,start_column=2,end_column=2)
        wst.merge_cells(start_row=1,end_row=1,start_column=3,end_column=7)
        wst.merge_cells(start_row=1,end_row=1,start_column=8,end_column=12)
        wst.merge_cells(start_row=1,end_row=1,start_column=13,end_column=16)
    wb.save(file)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter('{0}/inductor_log/{1} Dashboard Regression Check {0} {2}.xlsx'.format(args.target, args.backend, args.suite))
    generate_report(excel, args.reference, args.target)
    excel_postprocess(excel, 'Summary')
    excel_postprocess(excel, 'Summary New')
    html_generate(args.html_off)     
    update_issue_commits(args.precision)
