import argparse
from ast import arg
import pandas as pd
import json
import os
from datetime import datetime,timedelta
from styleframe import StyleFrame, Styler, utils
from scipy.stats import gmean
import numpy as np

parser = argparse.ArgumentParser(description="Generate quant per report")
parser.add_argument('-t', '--target', type=str, help='target log file')
parser.add_argument('-r', '--refer', type=str, help='refer log file')
parser.add_argument('-l', '--url', type=str, help='jenkins job build url')
args=parser.parse_args()
new_performance_regression=pd.DataFrame()
dynamic_performance_regression=pd.DataFrame()
new_acc_regression=pd.DataFrame()
dynamic_acc_regression=pd.DataFrame()

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                perf_path=os.path.join(root, d)
                for file in os.listdir(perf_path):
                    if file.endswith('.json'):
                        return os.path.join(perf_path, file)
                
def get_acc_log(round,thread):
    for root, dirs, files in os.walk(round):
        for f in files:
            if thread in (os.path.join(root, f)):
                return os.path.join(root, f)       

def json2df(perf):
    with open(perf, 'r') as file:
        data = json.load(file)
    metrics = data['metrics']
    df = pd.DataFrame(list(metrics.items()), columns=['model', 'throughput'])
    df.sort_values(by=['model'], key=lambda col: col.str.lower(),inplace=True)
    return df

def log2df(acc):
    model = []
    accuracy = []
    with open(acc, 'r') as file:
        for line in file:
            if 'int8:' in line or 'fp32' in line:
                model.append(line.split(' ')[0])
                accuracy.append(float(line.split(' ')[5]))
    model_acc = {'model':model, 'accuracy':accuracy}
    model_acc = pd.DataFrame(model_acc)
    model_acc.sort_values(by=['model'], key=lambda col: col.str.lower(),inplace=True)
    return model_acc

def update_new_perfor_regression(df_summary, col):
    global new_performance_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.9)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    new_performance_regression = pd.concat([new_performance_regression,regression])
    new_performance_regression = new_performance_regression.drop_duplicates()

def update_dynamic_perfor_regression(df_summary, col):
    global dynamic_performance_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.9)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    dynamic_performance_regression = pd.concat([dynamic_performance_regression,regression])
    dynamic_performance_regression = dynamic_performance_regression.drop_duplicates()

def update_new_acc_regression(df_summary, col):
    global new_acc_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.95)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    new_acc_regression = pd.concat([new_acc_regression,regression])
    new_acc_regression = new_acc_regression.drop_duplicates()

def update_dynamic_acc_regression(df_summary, col):
    global dynamic_acc_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.95)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    dynamic_acc_regression = pd.concat([dynamic_acc_regression,regression])
    dynamic_acc_regression = dynamic_acc_regression.drop_duplicates()

def update_swinfo(excel):
    data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata','dynamo_benchmarks'], 'Branch':['nightly', 'chuanqiw/inductor_quant', 'nightly', 'nightly', 'nightly', 'nightly', 'nightly'], 'Target commit':[' ', ' ', ' ', ' ',' ',' ',' ']}
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

        swinfo.loc[0,"Target commit"]=torch_commit
        swinfo.loc[1,"Target commit"]=torchbench_commit
        swinfo.loc[2,"Target commit"]=torchaudio_commit
        swinfo.loc[3,"Target commit"]=torchtext_commit
        swinfo.loc[4,"Target commit"]=torchvision_commit
        swinfo.loc[5,"Target commit"]=torchdata_commit
        swinfo.loc[6,"Target commit"]=dynamo_benchmarks_commit

        # torch_main_commit=get_main_commit("pytorch",torch_commit)
        # torchaudio_main_commit=get_main_commit("audio",torchaudio_commit)
        # torchtext_main_commit=get_main_commit("text",torchtext_commit)
        # torchvision_main_commit=get_main_commit("vision",torchvision_commit)
        # torchdata_main_commit=get_main_commit("data",torchdata_commit) 

        # swinfo.loc[0,"Main commit"]=torch_main_commit
        # swinfo.loc[2,"Main commit"]=torchaudio_main_commit
        # swinfo.loc[3,"Main commit"]=torchtext_main_commit
        # swinfo.loc[4,"Main commit"]=torchvision_main_commit
        # swinfo.loc[5,"Main commit"]=torchdata_main_commit
        if args.refer is not None:
            data = {'SW':['Pytorch', 'Torchbench', 'torchaudio', 'torchtext','torchvision','torchdata','dynamo_benchmarks'], 'Refer commit':[' ', ' ', ' ', ' ',' ',' ',' ']}
            refer_version=pd.DataFrame(data)
            refer_swinfo = pd.read_table(args.refer+'/inductor_log/version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
            refer_torch_commit=refer_swinfo.loc[ 1, "commit"][-7:]
            refer_torchbench_commit=refer_swinfo.loc[ 0, "commit"][-8:]
            refer_torchaudio_commit=refer_swinfo.loc[ 4, "commit"][-7:]
            refer_torchtext_commit=refer_swinfo.loc[ 3, "commit"][-7:]
            refer_torchvision_commit=refer_swinfo.loc[ 2, "commit"][-7:]
            refer_torchdata_commit=refer_swinfo.loc[ 5, "commit"][-7:]
            refer_dynamo_benchmarks_commit=refer_swinfo.loc[ 6, "commit"][-7:]

            refer_version.loc[0,"Refer commit"]=refer_torch_commit
            refer_version.loc[1,"Refer commit"]=refer_torchbench_commit
            refer_version.loc[2,"Refer commit"]=refer_torchaudio_commit
            refer_version.loc[3,"Refer commit"]=refer_torchtext_commit
            refer_version.loc[4,"Refer commit"]=refer_torchvision_commit
            refer_version.loc[5,"Refer commit"]=refer_torchdata_commit
            refer_version.loc[6,"Refer commit"]=refer_dynamo_benchmarks_commit
            swinfo = pd.merge(swinfo, refer_version)
    except :
        print("version.txt not found")
        pass

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)   

def process_perf(excel, target, refer):
    ptq_perf = getfolder(target,'ptq')
    ptq_cpp_perf = getfolder(target,'cpp')
    qat_perf = getfolder(target,'qat')
    inductor_perf = getfolder(target,'general')

    ptq_ref = getfolder(refer,'ptq')
    ptq_cpp_ref = getfolder(refer,'cpp')
    qat_ref = getfolder(refer,'qat')
    inductor_ref = getfolder(refer,'general')

    ptq_df = json2df(ptq_perf)
    ptq_cpp_df = json2df(ptq_cpp_perf)
    qat_df = json2df(qat_perf)
    inductor_df = json2df(inductor_perf)

    ptq_df_ref = json2df(ptq_ref)
    ptq_cpp_df_ref = json2df(ptq_cpp_ref)
    qat_df_ref = json2df(qat_ref)
    inductor_df_ref = json2df(inductor_ref)

    df_summary = pd.DataFrame({
        'model_name': list(ptq_df['model']),
        'ptq_new': list(ptq_df['throughput']),
        'ptq_cpp_new': list(ptq_cpp_df['throughput']),
        'qat_new': list(qat_df['throughput']),
        'inductor_new': list(inductor_df['throughput']),
        'ptq_old': list(ptq_df_ref['throughput']),
        'ptq_cpp_old': list(ptq_cpp_df_ref['throughput']),
        'qat_old': list(qat_df_ref['throughput']),
        'inductor_old': list(inductor_df_ref['throughput'])
    })
    df_summary['ptq_cpp/ptq(new)']=pd.DataFrame(round(df_summary['ptq_cpp_new'] / df_summary['ptq_new'],2))
    df_summary['qat/ptq(new)']=pd.DataFrame(round(df_summary['qat_new'] / df_summary['ptq_new'],2))
    df_summary['ptq ratio(new/old)']=pd.DataFrame(round(df_summary['ptq_new'] / df_summary['ptq_old'],2))
    df_summary['ptq_cpp ratio(new/old)']=pd.DataFrame(round(df_summary['ptq_cpp_new'] / df_summary['ptq_cpp_old'],2))
    df_summary['qat ratio(new/old)']=pd.DataFrame(round(df_summary['qat_new'] / df_summary['qat_old'],2))
    df_summary['inductor ratio(new/old)']=pd.DataFrame(round(df_summary['inductor_new'] / df_summary['inductor_old'],2))
    
    quant_data = {'Perf_Geomean':['PTQ_CPP_WRAPPER/PTQ', 'QAT/PTQ'], 'Ratio':[' ', ' ']}
    perf_gm=pd.DataFrame(quant_data)
    perf_gm.loc[0,"Ratio"]=f"{gmean(round(df_summary['ptq_cpp_new'] / df_summary['ptq_new'],2)):.2f}x"
    perf_gm.loc[1,"Ratio"]=f"{gmean(round(df_summary['qat_new'] / df_summary['ptq_new'],2)):.2f}x"
    #df_summary.to_excel(str(args.name) + '.xlsx', index=False)
    df_summary=StyleFrame(df_summary)
    df_summary.set_column_width(1, 32)
    df_summary.set_column_width(2, 16)
    df_summary.set_column_width(3, 16)
    df_summary.set_column_width(4, 16)
    df_summary.set_column_width(5, 16)
    df_summary.set_column_width(6, 16)
    df_summary.set_column_width(7, 16)
    df_summary.set_column_width(8, 16)
    df_summary.set_column_width(9, 16)
    df_summary.set_column_width(10, 12)
    df_summary.set_column_width(11, 12)
    df_summary.set_column_width(12, 12)
    df_summary.set_column_width(13, 12)
    df_summary.set_column_width(14, 12)
    df_summary.set_column_width(15, 12)

    regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
    improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp/ptq(new)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat/ptq(new)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] < 0.9],styler_obj=regression_style)

    update_new_perfor_regression(df_summary,'ptq ratio(new/old)')
    update_new_perfor_regression(df_summary,'qat ratio(new/old)')
    update_new_perfor_regression(df_summary,'ptq_cpp ratio(new/old)')

    df_summary.to_excel(sheet_name='performance',excel_writer=excel)

    gm = StyleFrame(perf_gm)
    gm.set_column_width(1, 30)
    gm.set_column_width(2, 20)

    gm.to_excel(sheet_name='Perf Geomean',excel_writer=excel)

def process_acc(excel, target, refer):
    ptq_acc = get_acc_log(target,'acc_ptq.log') 
    ptq_cpp_acc = get_acc_log(target,'acc_ptq_cpp')
    qat_acc = get_acc_log(target,'acc_qat')
    fp32_acc = get_acc_log(target,'acc_fp32')

    ptq_acc_refer = get_acc_log(refer,'acc_ptq.log') 
    ptq_cpp_acc_refer = get_acc_log(refer,'acc_ptq_cpp')
    qat_acc_refer = get_acc_log(refer,'acc_qat')
    fp32_acc_refer = get_acc_log(refer,'acc_fp32')

    ptq_acc_df = log2df(ptq_acc)
    ptq_cpp_acc_df = log2df(ptq_cpp_acc)
    qat_acc_df = log2df(qat_acc)
    fp32_acc_df = log2df(fp32_acc)

    ptq_acc_df_ref = log2df(ptq_acc_refer)
    ptq_cpp_acc_df_ref = log2df(ptq_cpp_acc_refer)
    qat_acc_df_ref = log2df(qat_acc_refer)
    fp32_acc_df_ref = log2df(fp32_acc_refer)

    df_summary_acc=pd.DataFrame(ptq_acc_df.iloc[:,0])
    df_summary_acc.insert(loc=1, column='ptq_acc_new', value=ptq_acc_df.iloc[:,1])
    df_summary_acc.insert(loc=2, column='ptq_cpp_acc_new', value=ptq_cpp_acc_df.iloc[:,1])
    df_summary_acc.insert(loc=3, column='qat_acc_new', value=qat_acc_df.iloc[:,1])
    df_summary_acc.insert(loc=4, column='fp32_acc_new', value=fp32_acc_df.iloc[:,1])

    df_summary_acc.insert(loc=5, column='ptq_cpp/ptq(new)', value=round(ptq_cpp_acc_df.iloc[:,1] / ptq_acc_df.iloc[:,1],2))
    df_summary_acc.insert(loc=6, column='qat/ptq(new)', value=round(qat_acc_df.iloc[:,1] / ptq_acc_df.iloc[:,1],2))

    df_summary_acc.insert(loc=7, column='ptq_old', value=ptq_acc_df_ref.iloc[:,1])
    df_summary_acc.insert(loc=8, column='ptq_cpp_old', value=ptq_cpp_acc_df_ref.iloc[:,1])
    df_summary_acc.insert(loc=9, column='qat_old', value=qat_acc_df_ref.iloc[:,1])
    df_summary_acc.insert(loc=10, column='inductor_old', value=fp32_acc_df_ref.iloc[:,1])

    df_summary_acc.insert(loc=11, column='ptq ratio(new/old)', value=round(ptq_acc_df.iloc[:,1] / ptq_acc_df_ref.iloc[:,1],2))
    df_summary_acc.insert(loc=12, column='ptq_cpp ratio(new/old)', value=round(ptq_cpp_acc_df.iloc[:,1] / ptq_cpp_acc_df_ref.iloc[:,1],2))
    df_summary_acc.insert(loc=13, column='qat ratio(new/old)', value=round(qat_acc_df.iloc[:,1] / qat_acc_df_ref.iloc[:,1],2))
    df_summary_acc.insert(loc=14, column='inductor ratio(new/old)', value=round(fp32_acc_df.iloc[:,1] / fp32_acc_df_ref.iloc[:,1],2))

    df_summary=StyleFrame(df_summary_acc)
    df_summary.set_column_width(1, 32)
    df_summary.set_column_width(2, 16)
    df_summary.set_column_width(3, 16)
    df_summary.set_column_width(4, 16)
    df_summary.set_column_width(5, 16)
    df_summary.set_column_width(6, 16)
    df_summary.set_column_width(7, 16)
    df_summary.set_column_width(8, 16)
    df_summary.set_column_width(9, 16)
    df_summary.set_column_width(10, 12)
    df_summary.set_column_width(11, 12)
    df_summary.set_column_width(12, 12)
    df_summary.set_column_width(13, 12)
    df_summary.set_column_width(14, 12)
    df_summary.set_column_width(15, 12)

    regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
    improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp/ptq(new)'] < 0.95],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat/ptq(new)'] < 0.95],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] < 0.95],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] < 0.95],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] < 0.95],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] < 0.95],styler_obj=regression_style)

    update_new_acc_regression(df_summary,'ptq ratio(new/old)')
    update_new_acc_regression(df_summary,'qat ratio(new/old)')
    update_new_acc_regression(df_summary,'ptq_cpp ratio(new/old)')

    df_summary.to_excel(sheet_name='accuracy',excel_writer=excel)

    quant_data = {'ACC_Geomean':['PTQ_CPP_WRAPPER/PTQ', 'QAT/PTQ'], 'Ratio':[' ', ' ']}
    perf_gm=pd.DataFrame(quant_data)
    perf_gm.loc[0,"Ratio"]=f"{gmean(round(ptq_cpp_acc_df.iloc[:,1] / ptq_acc_df.iloc[:,1],2)):.2f}x"
    perf_gm.loc[1,"Ratio"]=f"{gmean(round(qat_acc_df.iloc[:,1] / ptq_acc_df.iloc[:,1],2)):.2f}x"

    gm = StyleFrame(perf_gm)
    gm.set_column_width(1, 30)
    gm.set_column_width(2, 20)

    gm.to_excel(sheet_name='ACC Geomean',excel_writer=excel)

def read_dynamic_log(log_path):
    model = []
    accuracy = []
    performance = []
    perf_data = 0
    accuracy_data = 0
    for root, dirs, files in os.walk(log_path):
        for f in files:
            file_path=os.path.join(log_path, f)
            model_name=f.split('-throughput-')[0]
            with open(file_path, 'r') as file:
                model.append(model_name)
                for line in file:
                    if '7/7' in line:
                        perf_data = line.split(' ')[-1].split('it/s')[0]
                        # print(perf_data)
                        # performance.append(float(line.split(' ')[-1].strip("\n")))
                        
                    if 'eval_accuracy' in line:
                        accuracy_data = line.split(' ')[-1].strip("\n")
                performance.append(float(perf_data))
                accuracy.append(float(accuracy_data))
    quant = {'model':model, 'eval_samples_per_second':performance, 'eval_accuracy':accuracy}
    quant = pd.DataFrame(quant)
    quant.sort_values(by=['model'], key=lambda col: col.str.lower(),inplace=True)
    return quant

def process_dynamic_perf(excel, target, refer):
    target_dynamic_quant_path = os.path.join(target, 'inductor_log', 'hf_quant', 'dynamic_quant')
    target_fp32_compile_path = os.path.join(target, 'inductor_log', 'hf_quant', 'fp32_compile')
    target_static_quant_path = os.path.join(target, 'inductor_log', 'hf_quant', 'static_quant')
    
    target_dynamic_quant = read_dynamic_log(target_dynamic_quant_path)
    target_fp32_compile = read_dynamic_log(target_fp32_compile_path)
    target_static_quant = read_dynamic_log(target_static_quant_path)

    refer_dynamic_quant_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'dynamic_quant')
    refer_fp32_compile_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'fp32_compile')
    refer_static_quant_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'static_quant')
    
    refer_dynamic_quant = read_dynamic_log(refer_dynamic_quant_path)
    refer_fp32_compile = read_dynamic_log(refer_fp32_compile_path)
    refer_static_quant = read_dynamic_log(refer_static_quant_path)

    perf_summary = pd.DataFrame({
        'model_name': list(target_dynamic_quant['model']),
        'fp32_compile_new': list(target_fp32_compile['eval_samples_per_second']),
        'static_quant_new': list(target_static_quant['eval_samples_per_second']),
        'dynamic_quant_new': list(target_dynamic_quant['eval_samples_per_second']),
        'fp32_compile_old': list(refer_fp32_compile['eval_samples_per_second']),
        'static_quant_old': list(refer_static_quant['eval_samples_per_second']),
        'dynamic_quant_old': list(refer_dynamic_quant['eval_samples_per_second']),
    })

    perf_summary['dynamic_quant/fp32_compile(new)']=pd.DataFrame(round(perf_summary['dynamic_quant_new'] / perf_summary['fp32_compile_new'],2))
    perf_summary['dynamic_quant/static_quant(new)']=pd.DataFrame(round(perf_summary['dynamic_quant_new'] / perf_summary['static_quant_new'],2))
    perf_summary['fp32_compile(new/old)']=pd.DataFrame(round(perf_summary['fp32_compile_new'] / perf_summary['fp32_compile_old'],2))
    perf_summary['static_quant(new/old)']=pd.DataFrame(round(perf_summary['static_quant_new'] / perf_summary['static_quant_old'],2))
    perf_summary['dynamic_quant(new/old)']=pd.DataFrame(round(perf_summary['dynamic_quant_new'] / perf_summary['dynamic_quant_old'],2))

    quant_data = {'Perf_Geomean':['dynamic_quant/fp32_compile', 'dynamic_quant/static_quant'], 'Ratio':[' ', ' ']}
    quant_perf_gm=pd.DataFrame(quant_data)
    quant_perf_gm.loc[0,"Ratio"]=f"{gmean(round(perf_summary['dynamic_quant_new'] / perf_summary['fp32_compile_new'],2)):.2f}x"
    quant_perf_gm.loc[1,"Ratio"]=f"{gmean(round(perf_summary['dynamic_quant_new'] / perf_summary['static_quant_new'],2)):.2f}x"
    #df_summary.to_excel(str(args.name) + '.xlsx', index=False)
    perf_summary=StyleFrame(perf_summary)
    perf_summary.set_column_width(1, 32)
    perf_summary.set_column_width(2, 20)
    perf_summary.set_column_width(3, 20)
    perf_summary.set_column_width(4, 20)
    perf_summary.set_column_width(5, 20)
    perf_summary.set_column_width(6, 20)
    perf_summary.set_column_width(7, 20)
    perf_summary.set_column_width(8, 20)
    perf_summary.set_column_width(9, 20)
    perf_summary.set_column_width(10, 20)
    perf_summary.set_column_width(11, 20)
    perf_summary.set_column_width(12, 20)

    regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
    improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)
    # perf_summary.apply_style_by_indexes(indexes_to_style=perf_summary[perf_summary['dynamic_quant/static_quant(new)'] < 0.9],styler_obj=regression_style)
    perf_summary.apply_style_by_indexes(indexes_to_style=perf_summary[perf_summary['static_quant(new/old)'] < 0.9],styler_obj=regression_style)
    perf_summary.apply_style_by_indexes(indexes_to_style=perf_summary[perf_summary['static_quant(new/old)'] > 1.1],styler_obj=improve_style)
    perf_summary.apply_style_by_indexes(indexes_to_style=perf_summary[perf_summary['dynamic_quant(new/old)'] > 1.1],styler_obj=improve_style)
    perf_summary.apply_style_by_indexes(indexes_to_style=perf_summary[perf_summary['dynamic_quant(new/old)'] < 0.9],styler_obj=regression_style)


    update_dynamic_perfor_regression(perf_summary,'static_quant(new/old)')
    update_dynamic_perfor_regression(perf_summary,'dynamic_quant(new/old)')

    perf_summary.to_excel(sheet_name='its_per_second',excel_writer=excel)

    gm = StyleFrame(quant_perf_gm)
    gm.set_column_width(1, 30)
    gm.set_column_width(2, 20)

    gm.to_excel(sheet_name='Dynamic Perf Geomean',excel_writer=excel)

def process_dynamic_acc(excel, target, refer):
    target_dynamic_quant_path = os.path.join(target, 'inductor_log', 'hf_quant', 'dynamic_quant')
    target_fp32_compile_path = os.path.join(target, 'inductor_log', 'hf_quant', 'fp32_compile')
    target_static_quant_path = os.path.join(target, 'inductor_log', 'hf_quant', 'static_quant')
    
    target_dynamic_quant = read_dynamic_log(target_dynamic_quant_path)
    target_fp32_compile = read_dynamic_log(target_fp32_compile_path)
    target_static_quant = read_dynamic_log(target_static_quant_path)

    refer_dynamic_quant_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'dynamic_quant')
    refer_fp32_compile_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'fp32_compile')
    refer_static_quant_path = os.path.join(refer, 'inductor_log', 'hf_quant', 'static_quant')
    
    refer_dynamic_quant = read_dynamic_log(refer_dynamic_quant_path)
    refer_fp32_compile = read_dynamic_log(refer_fp32_compile_path)
    refer_static_quant = read_dynamic_log(refer_static_quant_path)

    acc_summary = pd.DataFrame({
        'model_name': list(target_dynamic_quant['model']),
        'fp32_compile_new': list(target_fp32_compile['eval_accuracy']),
        'static_quant_new': list(target_static_quant['eval_accuracy']),
        'dynamic_quant_new': list(target_dynamic_quant['eval_accuracy']),
        'fp32_compile_old': list(refer_fp32_compile['eval_accuracy']),
        'static_quant_old': list(refer_static_quant['eval_accuracy']),
        'dynamic_quant_old': list(refer_dynamic_quant['eval_accuracy']),
    })

    acc_summary['dynamic_quant/fp32_compile(new)']=pd.DataFrame(round(acc_summary['dynamic_quant_new'] / acc_summary['fp32_compile_new'],2))
    acc_summary['dynamic_quant/static_quant(new)']=pd.DataFrame(round(acc_summary['dynamic_quant_new'] / acc_summary['static_quant_new'],2))
    acc_summary['fp32_compile(new/old)']=pd.DataFrame(round(acc_summary['fp32_compile_new'] / acc_summary['fp32_compile_old'],2))
    acc_summary['static_quant(new/old)']=pd.DataFrame(round(acc_summary['static_quant_new'] / acc_summary['static_quant_old'],2))
    acc_summary['dynamic_quant(new/old)']=pd.DataFrame(round(acc_summary['dynamic_quant_new'] / acc_summary['dynamic_quant_old'],2))

    quant_data = {'ACC_Geomean':['dynamic_quant/fp32_compile', 'dynamic_quant/static_quant'], 'Ratio':[' ', ' ']}
    quant_acc_gm=pd.DataFrame(quant_data)
    quant_acc_gm.loc[0,"Ratio"]=f"{gmean(round(acc_summary['dynamic_quant_new'] / acc_summary['fp32_compile_new'],2)):.2f}x"
    quant_acc_gm.loc[1,"Ratio"]=f"{gmean(round(acc_summary['dynamic_quant_new'] / acc_summary['static_quant_new'],2)):.2f}x"
    #df_summary.to_excel(str(args.name) + '.xlsx', index=False)
    acc_summary=StyleFrame(acc_summary)
    acc_summary.set_column_width(1, 32)
    acc_summary.set_column_width(2, 20)
    acc_summary.set_column_width(3, 20)
    acc_summary.set_column_width(4, 20)
    acc_summary.set_column_width(5, 20)
    acc_summary.set_column_width(6, 20)
    acc_summary.set_column_width(7, 20)
    acc_summary.set_column_width(8, 20)
    acc_summary.set_column_width(9, 20)
    acc_summary.set_column_width(10, 20)
    acc_summary.set_column_width(11, 20)
    acc_summary.set_column_width(12, 20)

    regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
    improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)
    # acc_summary.apply_style_by_indexes(indexes_to_style=acc_summary[acc_summary['dynamic_quant/static_quant(new)'] < 0.9],styler_obj=regression_style)
    acc_summary.apply_style_by_indexes(indexes_to_style=acc_summary[acc_summary['static_quant(new/old)'] < 0.9],styler_obj=regression_style)
    acc_summary.apply_style_by_indexes(indexes_to_style=acc_summary[acc_summary['static_quant(new/old)'] > 1.1],styler_obj=improve_style)
    acc_summary.apply_style_by_indexes(indexes_to_style=acc_summary[acc_summary['dynamic_quant(new/old)'] > 1.1],styler_obj=improve_style)
    acc_summary.apply_style_by_indexes(indexes_to_style=acc_summary[acc_summary['dynamic_quant(new/old)'] < 0.9],styler_obj=regression_style)


    update_dynamic_acc_regression(acc_summary,'static_quant(new/old)')
    update_dynamic_acc_regression(acc_summary,'dynamic_quant(new/old)')

    acc_summary.to_excel(sheet_name='eval_accuracy',excel_writer=excel)

    gm = StyleFrame(quant_acc_gm)
    gm.set_column_width(1, 30)
    gm.set_column_width(2, 20)

    gm.to_excel(sheet_name='Dynamic ACC Geomean',excel_writer=excel)

def html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<style type="text/css">
    table
    {
      border-collapse: collapse;
      margin: 0 auto;
    }
    table td, table th
    {
      border: 1px solid #cad9ea;
      color: #666;
      height: 30px;
    }
    table thead th
    {
      background-color: #CCE8EB;
      width: 100px;
    }
    table tr:nth-child(odd)
    {
      background: #fff;
    }
    table tr:nth-child(even)
    {
      background: #F5FAFA;
    }
  </style>
<head>
<title> Quantization Regular Model Bench Report </title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, follow">
</head>
<body>
  <div class="limiter">
  <div class="container-table100">
  <div class="wrap-table100">
  <div class="table100">
  <p><h3>Quantization Regular Model Bench Report </p></h3> '''

def ICX_info():
    return '''
<table width="90%">
    <thead>
        <tr>
            <th>Item</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Manufacturer</td>
            <td>Amazon EC2</td>
        </tr>
        <tr>
            <td>Product Name</td>
            <td>c6i.16xlarge</td>
        </tr>
        <tr>
            <td>CPU Model</td>
            <td>Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz</td>
        </tr>
        <tr>
            <td>Installed Memory</td>
            <td>128GB (1x128GB DDR4 3200 MT/s [3200 MT/s])</td>
        </tr>
        <tr>
            <td>OS</td>
            <td>Ubuntu 22.04.3 LTS</td>
        </tr>
        <tr>
            <td>Kernel</td>
            <td>6.2.0-1018-aws</td>
        </tr>
        <tr>
            <td>Microcode</td>
            <td>0xd0003d1</td>
        </tr>
        <tr>
            <td>GCC</td>
            <td>gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0</td>
        </tr>
        <tr>
            <td>GLIBC</td>
            <td>ldd (Ubuntu GLIBC 2.35-0ubuntu3.6) 2.35</td>
        </tr>
        <tr>
            <td>Binutils</td>
            <td>GNU ld (GNU Binutils for Ubuntu) 2.38</td>
        </tr>
        <tr>
            <td>Python</td>
            <td>Python 3.8.18</td>
        </tr>
        <tr>
            <td>OpenSSL</td>
            <td>OpenSSL 3.2.0 23 Nov 2023 (Library: OpenSSL 3.2.0 23 Nov 2023)</td>
        </tr>
    </tbody>
</table>'''

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

def html_generate():
    
    content = pd.read_excel(args.target+'/inductor_log/Quantization_Regression_Check_'+args.target+'.xlsx',sheet_name=[0,1,2,3,4,5,6,7,8])
    summary_perf= pd.DataFrame(content[1]).to_html(classes="table",index = False)
    summary_acc = pd.DataFrame(content[3]).to_html(classes="table",index = False)
    dynamic_perf = pd.DataFrame(content[5]).to_html(classes="table",index = False)
    dynamic_acc = pd.DataFrame(content[7]).to_html(classes="table",index = False)
    detail_static_perf = pd.DataFrame(content[0])
    static_perf_subset = ['ptq ratio(new/old)', 'ptq_cpp ratio(new/old)', 'qat ratio(new/old)']
    detail_static_perf = detail_static_perf.style.hide().\
        highlight_between(left=0,right=0.9,subset=static_perf_subset,props='color:black;background-color:#FF7A33').\
        highlight_between(left=1.1,right=float('inf'),subset=static_perf_subset,props='color:black;background-color:#ACFF33').to_html(classes="table",index = False)
    detail_dynamic_perf = pd.DataFrame(content[4])
    dynamic_perf_subset = ['static_quant(new/old)', 'dynamic_quant(new/old)']
    detail_dynamic_perf = detail_dynamic_perf.style.hide().\
        highlight_between(left=0,right=0.9,subset=dynamic_perf_subset,props='color:black;background-color:#FF7A33').\
        highlight_between(left=1.1,right=float('inf'),subset=dynamic_perf_subset,props='color:black;background-color:#ACFF33').to_html(classes="table",index = False)
    detail_static_acc = pd.DataFrame(content[2])
    static_acc_subset = ['ptq ratio(new/old)', 'ptq_cpp ratio(new/old)', 'qat ratio(new/old)']
    detail_static_acc = detail_static_acc.style.hide().\
        highlight_between(left=0,right=0.9,subset=static_acc_subset,props='color:black;background-color:#FF7A33').\
        highlight_between(left=1.1,right=float('inf'),subset=static_acc_subset,props='color:black;background-color:#ACFF33').to_html(classes="table",index = False)
    detail_dynamic_acc = pd.DataFrame(content[6])
    dynamic_acc_subset = ['static_quant(new/old)', 'dynamic_quant(new/old)']
    detail_dynamic_acc = detail_dynamic_acc.style.hide().\
        highlight_between(left=0,right=0.9,subset=dynamic_acc_subset,props='color:black;background-color:#FF7A33').\
        highlight_between(left=1.1,right=float('inf'),subset=dynamic_acc_subset,props='color:black;background-color:#ACFF33').to_html(classes="table",index = False)
    
    swinfo= pd.DataFrame(content[8]).to_html(classes="table",index = False)
    # refer_swinfo_html = ''
    # if args.refer is not None:
    #     refer_swinfo = pd.read_table(args.refer+'/inductor_log/version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
    #     refer_swinfo_html = refer_swinfo.to_html(classes="table",index = False)           
    perf_regression= new_performance_regression.to_html(classes="table",index = False)
    acc_regression= new_acc_regression.to_html(classes="table",index = False)
    dynamic_quant_perf_regression=dynamic_performance_regression.to_html(classes="table",index = False)
    dynamic_quant_acc_regression=dynamic_acc_regression.to_html(classes="table",index = False)
    with open(args.target+'/inductor_log/quantization_model_bench.html',mode = "a") as f,open(args.target+'/inductor_log/quantization_perf_regression.html',mode = "a") as perf_f:
        # f.write(html_head()+"<p>Hardware info</p>"+ICX_info()+"<p>SW info</p>"+swinfo+"<p>Static_Quant_Perf_Geomean</p>"+summary_perf+"<p>Static_Quant_ACC_Geomean</p>"+summary_acc+"<p>Dynamic_Quant_Perf_Geomean</p>"+dynamic_perf+"<p>Dynamic_Quant_ACC_Geomean</p>"+dynamic_acc+"<p>Static Quant Performance</p>"+detail_static_perf+"<p>Dynamic Quant Performance</p>"+detail_dynamic_perf+"<p>Static Quant Accuracy</p>"+detail_static_acc+"<p>Dynamic Quant Accuracy</p>"+detail_dynamic_acc+html_tail())
        f.write(html_head()+"<p>Hardware info</p>"+"<p>mlp-validate-icx</p>"+"<p>SW info</p>"+swinfo+"<p>Static_Quant_Perf_Geomean</p>"+summary_perf+"<p>Static_Quant_ACC_Geomean</p>"+summary_acc+"<p>Dynamic_Quant_Perf_Geomean</p>"+dynamic_perf+"<p>Dynamic_Quant_ACC_Geomean</p>"+dynamic_acc+"<p>Static Quant Performance</p>"+detail_static_perf+"<p>Dynamic Quant Performance</p>"+detail_dynamic_perf+"<p>Static Quant Accuracy</p>"+detail_static_acc+"<p>Dynamic Quant Accuracy</p>"+detail_dynamic_acc+html_tail())
        perf_f.write(f"<p>new_perf_regression in {str((datetime.now() - timedelta(days=2)).date())}</p>"+"<p>new_static_perf_regression</p>"+perf_regression+"<p>new_dynamic_perf_regression</p>"+dynamic_quant_perf_regression+"<p>SW info</p>"+swinfo+"<p>Reference SW info (nightly)</p>")
    f.close()
    perf_f.close()            
    
excel = StyleFrame.ExcelWriter(args.target+'/inductor_log/Quantization_Regression_Check_'+args.target+'.xlsx')
process_perf(excel, args.target, args.refer)
process_acc(excel, args.target, args.refer)
process_dynamic_perf(excel, args.target, args.refer)
process_dynamic_acc(excel, args.target, args.refer)
update_swinfo(excel)
excel.close()
html_generate()


