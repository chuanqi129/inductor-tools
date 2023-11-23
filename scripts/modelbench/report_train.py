"""
Generate training benchmark report from inductor training logs.
Usage:
  python report_train.py -t WW17.4 -r WW06.2
"""
import argparse
import pandas as pd
import os
import requests
from styleframe import StyleFrame, Styler, utils
from bs4 import BeautifulSoup


parser = argparse.ArgumentParser(description="Generate report from inductor training logs")
parser.add_argument('-t','--target',type=str,help='target log file')
parser.add_argument('-r','--reference',type=str,help='reference log file')
args=parser.parse_args()

passed_style = Styler(bg_color='#D8E4BC', font_color=utils.colors.black)
failed_style = Styler(bg_color='#FFC7CE', font_color=utils.colors.black)

red_style = Styler(bg_color='#FF0000', font_color=utils.colors.black)
regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)

def get_data_file(dir_name,option):
    path = dir_name+'/inductor_log/'
    for f in os.listdir(path):
        if f.endswith(option+'.csv'):
            data_path=path+f
    return data_path
target_perf_path=get_data_file(args.target,'perf')
target_acc_path=get_data_file(args.target,'acc')
if args.reference is not None:
    reference_perf_path=get_data_file(args.reference,'perf')
    reference_acc_path=get_data_file(args.reference,'acc')  

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
        swinfo.loc[0,"Nightly commit"]=version.loc[ 1, "commit"][-7:]
        swinfo.loc[1,"Main commit"]=version.loc[ 0, "commit"][-8:]
        swinfo.loc[2,"Nightly commit"]=version.loc[ 4, "commit"][-7:]
        swinfo.loc[3,"Nightly commit"]=version.loc[ 3, "commit"][-7:]
        swinfo.loc[4,"Nightly commit"]=version.loc[ 2, "commit"][-7:]
        swinfo.loc[5,"Nightly commit"]=version.loc[ 5, "commit"][-7:]
        swinfo.loc[6,"Nightly commit"]=version.loc[ 6, "commit"][-7:]

        # To do: Not only for pytorch nightly branch
        swinfo.loc[0,"Main commit"]=get_main_commit("pytorch",version.loc[ 1, "commit"][-7:])
        swinfo.loc[2,"Main commit"]=get_main_commit("audio",version.loc[ 4, "commit"][-7:])
        swinfo.loc[3,"Main commit"]=get_main_commit("text",version.loc[ 3, "commit"][-7:])
        swinfo.loc[4,"Main commit"]=get_main_commit("vision",version.loc[ 2, "commit"][-7:])
        swinfo.loc[5,"Main commit"]=get_main_commit("data",version.loc[ 5, "commit"][-7:]) 
    except :
        print("version.txt not found")
        pass

    sf = StyleFrame(swinfo)
    sf.set_column_width(1, 25)
    sf.set_column_width(2, 20)
    sf.set_column_width(3, 25)

    sf.to_excel(sheet_name='SW',excel_writer=excel)

def update_failures(excel):
    tmp=[]
    perf_data=pd.read_csv(target_perf_path)
    acc_data=pd.read_csv(target_acc_path)
    
    acc_data=acc_data.loc[(acc_data['accuracy'] =='fail_to_run') | (acc_data['accuracy'] =='fail_accuracy')| (acc_data['batch_size'] ==0),:]
    tmp.append(acc_data)

    perf_data=perf_data.loc[perf_data['speedup'] ==0,:]
    tmp.append(perf_data)

    failures=pd.concat(tmp)
    failures=failures[['name','accuracy','speedup']]

    failures['accuracy'].replace(["fail_to_run","fail_accuracy","0.0000"],["X","X","X"],inplace=True)
    failures['speedup'].replace([0],["X"],inplace=True)
    
    failures=failures.rename(columns={'name':'name','accuracy':'accuracy','speedup':'perf'})
    failures['perf'].replace([0],["√"],inplace=True)
    failures['accuracy'].replace([0],["√"],inplace=True)

    sf = StyleFrame({'name': list(failures['name']),
                 'accuracy': list(failures['accuracy']),
                 'perf': list(failures['perf'])})
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

    sf.to_excel(sheet_name='Failures',excel_writer=excel,index=False)

def process_suite():
    target_ori_data=pd.read_csv(target_perf_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency']]
    target_data=pd.DataFrame(target_data)
    if args.reference is not None:
        reference_ori_data=pd.read_csv(reference_perf_path,index_col=0)
        reference_data=reference_ori_data[['name','batch_size','speedup','abs_latency']]
        reference_data=pd.DataFrame(reference_data)
        data=pd.merge(target_data,reference_data,on=['name'],how= 'outer')
        return data
    return target_data
 
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
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] > 1.1],styler_obj=improve_style)
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
        data.set_row_height(rows=data.row_indexes, height=15)
    return data

def update_details(writer):
    h = {"A": 'Model suite',"B": '', "C": args.target, "D": '', "E": '',"F": '', "G": args.reference, "H": '', "I": '',"J": '',"K": 'Result Comp',"L": '',"M": ''}
    if args.reference is None:
        h = {"A": 'Model suite',"B": '', "C": args.target, "D": '', "E": '',"F": ''}    
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    head.to_excel(excel_writer=writer, sheet_name='Single-Socket Multi-threads', index=False,startrow=0,header=False)
    mt=process_suite()
    mt_data=process(mt)

    suite_list=[''] * 9
    suite_list.insert(0, 'Torchbench')
    suite_list.insert(2, 'HF')
    suite_list.insert(6, 'Timm')

    s= pd.Series(suite_list)
    s.to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=0)
    mt_data.to_excel(sheet_name='Single-Socket Multi-threads',excel_writer=writer,index=False,startrow=1,startcol=1)

def html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<title>Inductor Model Bench Report</title>
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
  <p><h3>Inductor Model Training Bench Report</p></h3> '''

def html_tail():
    # Use true HW info 
    return '''<p>You can find details from attachments, Thanks</p>
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
    try:
        content = pd.read_excel(args.target+'/inductor_log/inductor_model_training_check '+args.target+'.xlsx',sheet_name=[0,1,2])
        swinfo= pd.DataFrame(content[0]).to_html(classes="table",index = False)
        failures= pd.DataFrame(content[1]).to_html(classes="table",index = False)
        details= pd.DataFrame(content[2],index=range(11)).to_html(classes="table",index = False,header=False,na_rep='*')
        with open(args.target+'/inductor_log/inductor_model_training_bench.html',mode = "a") as f:
            f.write(html_head()+"<p>SW info</p>"+swinfo+"<p>Failures</p>"+failures+"<p>Details</p>"+details+html_tail())
        f.close()
    except:
        print("html_generate_failed")
        pass


def excel_postprocess(file):
    wb=file.book
    # Single-Socket Multi-threads
    wmt=wb['Single-Socket Multi-threads']
    wmt.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
    wmt.merge_cells(start_row=1,end_row=1,start_column=3,end_column=6)
    wmt.merge_cells(start_row=1,end_row=1,start_column=7,end_column=10)
    wmt.merge_cells(start_row=1,end_row=1,start_column=11,end_column=13)
    wmt.merge_cells(start_row=3,end_row=4,start_column=1,end_column=1)
    wmt.merge_cells(start_row=5,end_row=8,start_column=1,end_column=1)
    wmt.merge_cells(start_row=9,end_row=12,start_column=1,end_column=1)
    wb.save(file)

def generate_report(excel):
    update_swinfo(excel)
    update_failures(excel)
    update_details(excel)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter(args.target+'/inductor_log/inductor_model_training_check '+args.target+'.xlsx')
    generate_report(excel)
    excel_postprocess(excel)
    html_generate()