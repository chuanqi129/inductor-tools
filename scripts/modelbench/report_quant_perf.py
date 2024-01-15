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
new_acc_regression=pd.DataFrame()

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
            if 'int8' in line or 'fp32' in line:
                model.append(line.split(' ')[0])
                accuracy.append(float(line.split(' ')[5]))
    model_acc = {'model':model, 'accuracy':accuracy}
    model_acc = pd.DataFrame(model_acc)
    model_acc.sort_values(by=['model'], key=lambda col: col.str.lower(),inplace=True)
    return model_acc

def update_new_perfer_regression(df_summary, col):
    global new_performance_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.9)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    new_performance_regression = pd.concat([new_performance_regression,regression])

def update_new_acc_regression(df_summary, col):
    global new_acc_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.99)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    new_acc_regression = pd.concat([new_acc_regression,regression])

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

    df_summary=pd.DataFrame(ptq_df.iloc[:,0])
    df_summary.insert(loc=1, column='ptq_new', value=ptq_df.iloc[:,1])
    df_summary.insert(loc=2, column='ptq_cpp_new', value=ptq_cpp_df.iloc[:,1])
    df_summary.insert(loc=3, column='qat_new', value=qat_df.iloc[:,1])
    df_summary.insert(loc=4, column='inductor_new', value=inductor_df.iloc[:,1])

    df_summary.insert(loc=5, column='ptq_cpp/ptq(new)', value=round(ptq_cpp_df.iloc[:,1] / ptq_df.iloc[:,1],2))
    df_summary.insert(loc=6, column='qat/ptq(new)', value=round(qat_df.iloc[:,1] / ptq_df.iloc[:,1],2))

    df_summary.insert(loc=7, column='ptq_old', value=ptq_df_ref.iloc[:,1])
    df_summary.insert(loc=8, column='ptq_cpp_old', value=ptq_cpp_df_ref.iloc[:,1])
    df_summary.insert(loc=9, column='qat_old', value=qat_df_ref.iloc[:,1])
    df_summary.insert(loc=10, column='inductor_old', value=inductor_df_ref.iloc[:,1])

    df_summary.insert(loc=11, column='ptq ratio(new/old)', value=round(ptq_df.iloc[:,1] / ptq_df_ref.iloc[:,1],2))
    df_summary.insert(loc=12, column='ptq_cpp ratio(new/old)', value=round(ptq_cpp_df.iloc[:,1] / ptq_cpp_df_ref.iloc[:,1],2))
    df_summary.insert(loc=13, column='qat ratio(new/old)', value=round(qat_df.iloc[:,1] / qat_df_ref.iloc[:,1],2))
    df_summary.insert(loc=14, column='inductor ratio(new/old)', value=round(inductor_df.iloc[:,1] / inductor_df_ref.iloc[:,1],2))

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

    update_new_perfer_regression(df_summary,'ptq ratio(new/old)')
    update_new_perfer_regression(df_summary,'qat ratio(new/old)')
    update_new_perfer_regression(df_summary,'ptq_cpp ratio(new/old)')

    df_summary.to_excel(sheet_name='performance',excel_writer=excel)

    quant_data = {'Perf_Geomean':['PTQ_CPP_WRAPPER/PTQ', 'QAT/PTQ'], 'Ratio':[' ', ' ']}
    perf_gm=pd.DataFrame(quant_data)
    perf_gm.loc[0,"Ratio"]=f"{gmean(round(ptq_cpp_df.iloc[:,1] / ptq_df.iloc[:,1],2)):.2f}x"
    perf_gm.loc[1,"Ratio"]=f"{gmean(round(qat_df.iloc[:,1] / ptq_df.iloc[:,1],2)):.2f}x"

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
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp/ptq(new)'] < 0.99],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat/ptq(new)'] < 0.99],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq ratio(new/old)'] < 0.99],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['ptq_cpp ratio(new/old)'] < 0.99],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['qat ratio(new/old)'] < 0.99],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] > 1.1],styler_obj=improve_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['inductor ratio(new/old)'] < 0.99],styler_obj=regression_style)

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

def html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<title> Quantization Regular Model Bench Report </title>
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
  <p><h3>Quantization Regular Model Bench Report </p></h3> '''

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
    
    content = pd.read_excel(args.target+'/inductor_log/Quantization_Regression_Check_'+args.target+'.xlsx',sheet_name=[0,1,2,3,4])
    summary_perf= pd.DataFrame(content[1]).to_html(classes="table",index = False)
    summary_acc = pd.DataFrame(content[3]).to_html(classes="table",index = False)
    swinfo= pd.DataFrame(content[4]).to_html(classes="table",index = False)
    refer_swinfo_html = ''
    if args.refer is not None:
        refer_swinfo = pd.read_table(args.refer+'/inductor_log/version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
        refer_swinfo_html = refer_swinfo.to_html(classes="table",index = False)            
    perf_regression= new_performance_regression.to_html(classes="table",index = False)
    acc_regression= new_acc_regression.to_html(classes="table",index = False)
    with open(args.target+'/inductor_log/quantization_model_bench.html',mode = "a") as f,open(args.target+'/inductor_log/quantization_perf_regression.html',mode = "a") as perf_f:
        f.write(html_head()+"<p>Summary_Perf</p>"+summary_perf+"<p>Summary_ACC</p>"+summary_acc+"<p>SW info</p>"+swinfo+"<p>new_perf_regression</p>"+perf_regression+"<p>new_acc_regression</p>"+acc_regression+html_tail())
        perf_f.write(f"<p>new_perf_regression in {str((datetime.now() - timedelta(days=2)).date())}</p>"+perf_regression+"<p>SW info</p>"+swinfo+"<p>Reference SW info (nightly)</p>"+refer_swinfo_html)
    f.close()
    perf_f.close()            
    
excel = StyleFrame.ExcelWriter(args.target+'/inductor_log/Quantization_Regression_Check_'+args.target+'.xlsx')
process_perf(excel, args.target, args.refer)
process_acc(excel, args.target, args.refer)
update_swinfo(excel)
excel.close()
html_generate()


