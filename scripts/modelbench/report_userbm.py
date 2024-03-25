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

def getfolder(round,thread):
    for root, dirs, files in os.walk(round):
        for d in dirs:
            if thread in (os.path.join(root, d)):
                perf_path=os.path.join(root, d)
                for file in os.listdir(perf_path):
                    if file.endswith('.json'):
                        return os.path.join(perf_path, file)    

def json2df(perf):
    with open(perf, 'r') as file:
        data = json.load(file)
    metrics = data['metrics']
    df = pd.DataFrame(list(metrics.items()), columns=['model', 'throughput'])
    df.sort_values(by=['model'], key=lambda col: col.str.lower(),inplace=True)
    return df

def update_new_perfer_regression(df_summary, col):
    global new_performance_regression
    regression = df_summary.loc[(df_summary[col] > 0) & (df_summary[col] < 0.9)]
    regression = regression.copy()
    regression.loc[0] = list(regression.shape[1]*'*')
    new_performance_regression = pd.concat([new_performance_regression,regression])
    new_performance_regression = new_performance_regression.drop_duplicates()

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

    except :
        print("version.txt not found")
        pass

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)   

def process_perf(excel, target, refer, suite):
    usebm_perf = getfolder(target, suite)
    usebm_ref = getfolder(refer,suite)

    usebm_df = json2df(usebm_perf)
    usebm_df_ref = json2df(usebm_ref)

    # df_summary=pd.DataFrame(usebm_df.iloc[:,0])
    # df_summary.insert(loc=1, column='throughput_new', value=usebm_df.iloc[:,1])
    # df_summary.insert(loc=2, column='thrpughput_old', value=usebm_df_ref.iloc[:,1])
    # df_summary.insert(loc=3, column='throughput ratio(new/old)', value=round(usebm_df.iloc[:,1] / usebm_df_ref.iloc[:,1],2))
    usebm_df['throughput'] = usebm_df['throughput'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    usebm_df_ref['throughput'] =usebm_df_ref['throughput'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_summary = pd.DataFrame({
        'model_name': list(usebm_df['model']),
        'throughput_new': list(usebm_df['throughput']),
        'throughput_old': list(usebm_df_ref['throughput'])
    })
    df_summary['throughput ratio(new/old)']=pd.DataFrame(round(df_summary['throughput_new'] / df_summary['throughput_old'],2))
    #df_summary.to_excel(str(args.name) + '.xlsx', index=False)
    geomean = f"{gmean(round(df_summary['throughput_new'] / df_summary['throughput_old'],2)):.2f}x"
    df_summary=StyleFrame(df_summary)
    df_summary.set_column_width(1, 32)
    df_summary.set_column_width(2, 32)
    df_summary.set_column_width(3, 32)
    df_summary.set_column_width(4, 32)

    regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
    improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['throughput ratio(new/old)'] < 0.9],styler_obj=regression_style)
    df_summary.apply_style_by_indexes(indexes_to_style=df_summary[df_summary['throughput ratio(new/old)'] > 1.1],styler_obj=improve_style)

    update_new_perfer_regression(df_summary,'throughput ratio(new/old)')

    df_summary.to_excel(sheet_name=suite, excel_writer=excel)

    return geomean

def html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<title> Userbenchmark Regular Model Bench Report </title>
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
    
    content = pd.read_excel(args.target+'/inductor_log/Userbenchmark_Regression_Check_'+args.target+'.xlsx',sheet_name=[0,1,2,3,4,5,6,7,8])
    summary_perf= pd.DataFrame(content[8]).to_html(classes="table",index = False)
    swinfo= pd.DataFrame(content[0]).to_html(classes="table",index = False)
    refer_swinfo_html = ''
    if args.refer is not None:
        refer_swinfo = pd.read_table(args.refer+'/inductor_log/version.txt', sep = '\:', header = None,names=['item', 'commit'],engine='python')
        refer_swinfo_html = refer_swinfo.to_html(classes="table",index = False)            
    perf_regression= new_performance_regression.to_html(classes="table",index = False)
    with open(args.target+'/inductor_log/userbenchmark_model_bench.html',mode = "a") as f:
        f.write(html_head()+"<p>Summary_Perf</p>"+summary_perf+"<p>SW info</p>"+swinfo+"<p>Reference SW info (nightly)</p>"+refer_swinfo_html+"<p>new_perf_regression</p>"+perf_regression+html_tail())
    f.close()        
    
excel = StyleFrame.ExcelWriter(args.target+'/inductor_log/Userbenchmark_Regression_Check_'+args.target+'.xlsx')
update_swinfo(excel)
eager_throughtput_bf16_infer = process_perf(excel, args.target, args.refer, "eager_throughtput_bf16_infer")
eager_throughtput_fp32_train = process_perf(excel, args.target, args.refer, "eager_throughtput_fp32_train")
jit_llga_throughtput_fp32 = process_perf(excel, args.target, args.refer, "jit_llga_throughtput_fp32")
eager_throughtput_bf16_train = process_perf(excel, args.target, args.refer, "eager_throughtput_bf16_train")
eager_throughtput_fx_int8 = process_perf(excel, args.target, args.refer, "eager_throughtput_fx_int8")
eager_throughtput_fp32_infer = process_perf(excel, args.target, args.refer, "eager_throughtput_fp32_infer")
jit_llga_throughtput_amp_bf16 = process_perf(excel, args.target, args.refer, "jit_llga_throughtput_amp_bf16")
usebm_ratio = {'Perf_Geomean':['eager_throughtput_bf16_infer', 'eager_throughtput_fp32_infer', 'jit_llga_throughtput_amp_bf16', 'jit_llga_throughtput_fp32', 'eager_throughtput_fx_int8', 'eager_throughtput_bf16_train', 'eager_throughtput_fp32_train'], 'Ratio (new/old)':[' ', ' ', ' ', ' ', ' ', ' ', ' ']}
userbm_gm=pd.DataFrame(usebm_ratio)
userbm_gm.loc[0,"Ratio (new/old)"]=eager_throughtput_bf16_infer
userbm_gm.loc[1,"Ratio (new/old)"]=eager_throughtput_fp32_infer
userbm_gm.loc[2,"Ratio (new/old)"]=jit_llga_throughtput_amp_bf16
userbm_gm.loc[3,"Ratio (new/old)"]=jit_llga_throughtput_fp32
userbm_gm.loc[4,"Ratio (new/old)"]=eager_throughtput_fx_int8
userbm_gm.loc[5,"Ratio (new/old)"]=eager_throughtput_bf16_train
userbm_gm.loc[6,"Ratio (new/old)"]=eager_throughtput_fp32_train
userbm_gm = StyleFrame(userbm_gm)
userbm_gm.set_column_width(1, 36)
userbm_gm.set_column_width(2, 20)
userbm_gm.to_excel(sheet_name='Perf Geomean',excel_writer=excel)
excel.close()
html_generate()


