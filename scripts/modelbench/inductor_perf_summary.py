import argparse

import pandas as pd
from scipy.stats import gmean
from styleframe import StyleFrame, Styler, utils

parser = argparse.ArgumentParser(description="Generate report")
parser.add_argument('-s', '--suite', default='huggingface', choices=["torchbench", "huggingface", "timm_models"], type=str, help='model suite name')
parser.add_argument('-p', '--precision', default='amp_bf16 amp_fp16', nargs='*', type=str, help='precision')
parser.add_argument('-r', '--reference', type=str, help='reference log files')
args = parser.parse_args()

passrate_values = {}
geomean_values = {}

new_performance_regression=pd.DataFrame()

failure_style = Styler(bg_color='#FF0000', font_color=utils.colors.black)
regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)

# refer https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/runner.py#L757-L778


def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)


def get_passing_entries(df, column_name):
    return df[column_name][df[column_name] > 0]


def caculate_geomean(df, column_name):
    cleaned_df = get_passing_entries(df, column_name).clip(1)
    if cleaned_df.empty:
        return "0.0x"
    return f"{gmean(cleaned_df):.2f}x"


def caculate_passrate(df, compiler):
    total = len(df.index)
    passing = df[df[compiler] > 0.0][compiler].count()
    perc = int(percentage(passing, total, decimals=0))
    return f"{perc}%, {passing}/{total}"


def get_perf_csv(precision, mode):
    target_path = 'inductor_log/' + args.suite + '/' + precision + '/inductor_' + args.suite + '_' + precision + '_' + mode + '_xpu_performance.csv'
    target_ori_data = pd.read_csv(target_path, header=0, encoding='utf-8', names=['dev', 'name', 'batch_size', 'speedup', 'abs_latency', 'compilation_latency', 'compression_ratio'])
    target_data = target_ori_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(), inplace=True)
    
    if args.reference is not None:
        reference_file_path = args.reference + '/inductor_log/' + args.suite + '/' + precision + '/inductor_' + args.suite + '_' + precision + '_' + mode + '_xpu_performance.csv'
        reference_ori_data = pd.read_csv(reference_file_path, header=0, encoding='utf-8', names=['dev', 'name', 'batch_size', 'speedup', 'abs_latency', 'compilation_latency', 'compression_ratio'])
        reference_data = reference_ori_data.copy()
        reference_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)    
        data = pd.merge(target_data,reference_data,on=['name'],how= 'outer')
        return data
    else:
        return target_data


def process(input, precision, mode):
    global geomean_values, passrate_values
    if args.reference is None:
        data_new = input[['name', 'batch_size', 'speedup', 'abs_latency','compilation_latency']].rename(columns={'name': 'name', 'batch_size': 'batch_size', 'speedup': 'speedup', "abs_latency": 'inductor', "compilation_latency": 'compilation_latency'})
        data_new['inductor'] = data_new['inductor'].apply(pd.to_numeric, errors='coerce').div(1000)
        data_new['speedup'] = data_new['speedup'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager'] = data_new['speedup'] * data_new['inductor']
        geomean_values['target_' + str(precision) + '_' + str(mode)] = caculate_geomean(data_new, 'speedup')
        passrate_values['target_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_new, 'inductor')
        data = StyleFrame({'name': list(data_new['name']),
                        'batch_size': list(data_new['batch_size']),
                        'speedup': list(data_new['speedup']),
                        'inductor': list(data_new['inductor']),
                        'eager': list(data_new['eager']),
                        'compilation_latency': list(data_new['compilation_latency'])})
        data.set_column_width(1, 10)
        data.set_column_width(2, 18)
        data.set_column_width(3, 18)
        data.set_column_width(4, 18)
        data.set_column_width(5, 15)
        data.set_column_width(6, 20)
        data.set_row_height(rows=data.row_indexes, height=15)
    else:
        data_new=input[['name','batch_size_x','speedup_x','abs_latency_x','compilation_latency_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','speedup_x':'speed_up_new',"abs_latency_x":'inductor_new',"compilation_latency_x":'compilation_latency_new'})
        data_new['inductor_new']=data_new['inductor_new'].astype(float).div(1000)
        data_new['speed_up_new']=data_new['speed_up_new'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_new['eager_new'] = data_new['speed_up_new'] * data_new['inductor_new']
        geomean_values['target_' + str(precision) + '_' + str(mode)] = caculate_geomean(data_new, 'speed_up_new')
        passrate_values['target_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_new, 'inductor_new')        
        data_old=input[['batch_size_y','speedup_y','abs_latency_y','compilation_latency_y']].rename(columns={'batch_size_y':'batch_size_old','speedup_y':'speed_up_old',"abs_latency_y":'inductor_old',"compilation_latency_y":'compilation_latency_old'}) 
        data_old['inductor_old']=data_old['inductor_old'].astype(float).div(1000)
        data_old['speed_up_old']=data_old['speed_up_old'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        data_old['eager_old'] = data_old['speed_up_old'] * data_old['inductor_old']
        input['speedup_x']=input['speedup_x'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        input['speedup_y']=input['speedup_y'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        geomean_values['reference_' + str(precision) + '_' + str(mode)] = caculate_geomean(data_old, 'speed_up_old')
        passrate_values['reference_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_old, 'inductor_old')        
        data_ratio= pd.DataFrame(round(input['speedup_x'] / input['speedup_y'],2),columns=['Ratio Speedup(New/old)'])
        data_ratio['Eager Ratio(old/new)'] = pd.DataFrame(round(data_old['eager_old'] / data_new['eager_new'],2))
        data_ratio['Inductor Ratio(old/new)'] = pd.DataFrame(round(data_old['inductor_old'] / data_new['inductor_new'],2))
        data_ratio['Compilation_latency_Ratio(old/new)'] = pd.DataFrame(round(data_old['compilation_latency_old'] / data_new['compilation_latency_new'],2))
        
        combined_data = pd.DataFrame({
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
        data = StyleFrame(combined_data)
        data.set_column_width(1, 10)
        data.set_column_width(2, 18) 
        data.set_column_width(3, 18) 
        data.set_column_width(4, 18)
        data.set_column_width(5, 15)
        data.set_column_width(6, 20)
        data.set_column_width(7, 18)
        data.set_column_width(8, 18) 
        data.set_column_width(9, 18) 
        data.set_column_width(10, 15)
        data.set_column_width(11, 20)
        data.set_column_width(12, 28)
        data.set_column_width(13, 28) 
        data.set_column_width(14, 28)
        data.set_column_width(15, 32)
        data.apply_style_by_indexes(indexes_to_style=data[data['batch_size_new'] == 0], styler_obj=failure_style)
        data.apply_style_by_indexes(indexes_to_style=data[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < 0.9)],styler_obj=regression_style)
        global new_performance_regression
        regression = data.loc[(data['Inductor Ratio(old/new)'] > 0) & (data['Inductor Ratio(old/new)'] < 0.9)]
        regression = regression.copy()
        regression.loc[0] = list(regression.shape[1]*'*')
        new_performance_regression = pd.concat([new_performance_regression,regression])
        data.apply_style_by_indexes(indexes_to_style=data[data['Inductor Ratio(old/new)'] > 1.1],styler_obj=improve_style)
        data.set_row_height(rows=data.row_indexes, height=15)        
    return data


def update_details(precision, mode, excel):
    h = {"A": 'Model suite', "B": '', "C": "target", "D": '', "E": '', "F": '', "G": '',"H": args.reference, "I": '', "J": '',"K": '',"L":'',"M": 'Result Comp',"N": '',"O": '',"P":''}
    if args.reference is None:
        h = {"A": 'Model suite', "B": '', "C": "target", "D": '', "E": '', "F": '', "G": ''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    head.to_excel(excel_writer=excel, sheet_name=precision + '_' + mode, index=False, startrow=0, header=False)
    target_raw_data = get_perf_csv(precision, mode)
    target_data = process(target_raw_data, precision, mode)
    target_data.to_excel(excel_writer=excel, sheet_name=precision + '_' + mode, index=False, startrow=1, startcol=1)


def update_summary(excel):
    data = {
        'Test Scenario': ['AMP_BF16 Inference', ' ', 'AMP_BF16 Training', ' ', 'AMP_FP16 Inference', ' ', 'AMP_FP16 Training', ' ', 'BF16 Inference', ' ', 'BF16 Training', ' ', 'FP16 Inference', ' ', 'FP16 Training', ' ', 'FP32 Inference', ' ', 'FP32 Training', ' '],
        'Comp Item': ['Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup', 'Pass Rate', 'Geomean Speedup'],
        'Date': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'Compiler': ['inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor'],
        'torchbench': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'huggingface': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'timm_models ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'ref_torchbench ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'refer_huggingface ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'refer_timm_models ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    }
    summary = pd.DataFrame(data)
    # passrate
    summary.iloc[0:1, 5:6] = passrate_values['target_amp_bf16_inference']
    summary.iloc[2:3, 5:6] = passrate_values['target_amp_bf16_training']
    summary.iloc[4:5, 5:6] = passrate_values['target_amp_fp16_inference']
    summary.iloc[6:7, 5:6] = passrate_values['target_amp_fp16_training']
    # geomean_speedup
    summary.iloc[1:2, 5:6] = geomean_values['target_amp_bf16_inference']
    summary.iloc[3:4, 5:6] = geomean_values['target_amp_bf16_training']
    summary.iloc[5:6, 5:6] = geomean_values['target_amp_fp16_inference']
    summary.iloc[7:8, 5:6] = geomean_values['target_amp_fp16_training']

    if 'bfloat16' in args.precision:
        summary.iloc[8:9, 5:6] = passrate_values['target_bfloat16_inference']
        summary.iloc[10:11, 5:6] = passrate_values['target_bfloat16_training']
        summary.iloc[9:10, 5:6] = geomean_values['target_bfloat16_inference']
        summary.iloc[11:12, 5:6] = geomean_values['target_bfloat16_training']
    if 'float16' in args.precision:
        summary.iloc[12:13, 5:6] = passrate_values['target_float16_inference']
        summary.iloc[14:15, 5:6] = passrate_values['target_float16_training']
        summary.iloc[13:14, 5:6] = geomean_values['target_float16_inference']
        summary.iloc[15:16, 5:6] = geomean_values['target_float16_training']
    if 'float32' in args.precision:
        summary.iloc[16:17, 5:6] = passrate_values['target_float32_inference']
        summary.iloc[18:19, 5:6] = passrate_values['target_float32_training']
        summary.iloc[17:18, 5:6] = geomean_values['target_float32_inference']
        summary.iloc[19:20, 5:6] = geomean_values['target_float32_training']

    if args.reference is not None:
        # passrate
        summary.iloc[0:1, 8:9] = passrate_values['reference_amp_bf16_inference']
        summary.iloc[2:3, 8:9] = passrate_values['reference_amp_bf16_training']
        summary.iloc[4:5, 8:9] = passrate_values['reference_amp_fp16_inference']
        summary.iloc[6:7, 8:9] = passrate_values['reference_amp_fp16_training']
        # geomean_speedup
        summary.iloc[1:2, 8:9] = geomean_values['reference_amp_bf16_inference']
        summary.iloc[3:4, 8:9] = geomean_values['reference_amp_bf16_training']
        summary.iloc[5:6, 8:9] = geomean_values['reference_amp_fp16_inference']
        summary.iloc[7:8, 8:9] = geomean_values['reference_amp_fp16_training']

        if 'bfloat16' in args.precision:
            summary.iloc[8:9, 8:9] = passrate_values['reference_bfloat16_inference']
            summary.iloc[10:11, 8:9] = passrate_values['reference_bfloat16_training']
            summary.iloc[9:10, 8:9] = geomean_values['reference_bfloat16_inference']
            summary.iloc[11:12, 8:9] = geomean_values['reference_bfloat16_training']
        if 'float16' in args.precision:
            summary.iloc[12:13, 8:9] = passrate_values['reference_float16_inference']
            summary.iloc[14:15, 8:9] = passrate_values['reference_float16_training']
            summary.iloc[13:14, 8:9] = geomean_values['reference_float16_inference']
            summary.iloc[15:16, 8:9] = geomean_values['reference_float16_training']
        if 'float32' in args.precision:
            summary.iloc[16:17, 8:9] = passrate_values['reference_float32_inference']
            summary.iloc[18:19, 8:9] = passrate_values['reference_float32_training']
            summary.iloc[17:18, 8:9] = geomean_values['reference_float32_inference']
            summary.iloc[19:20, 8:9] = geomean_values['reference_float32_training']
    
    print("====================Summary=============================")
    print(summary)
    
    sf = StyleFrame(summary)
    for i in range(1, 11):
        sf.set_column_width(i, 22)
    for j in range(1, 22):
        sf.set_row_height(j, 30)
    sf.to_excel(sheet_name='Summary', excel_writer=excel)


def generate_report(excel, precision_list):
    for p in precision_list:
        for m in ['inference', 'training']:
            update_details(p, m, excel)
    update_summary(excel)


def excel_postprocess(file, precison):
    wb = file.book
    # Summary
    ws = wb['Summary']
    for i in range(2, 21, 2):
        ws.merge_cells(start_row=i, end_row=i + 1, start_column=1, end_column=1)
    # details
    for p in precison:
        for m in ['inference', 'training']:
            wdt = wb[p + '_' + m]
            wdt.merge_cells(start_row=1, end_row=2, start_column=1, end_column=1)
            wdt.merge_cells(start_row=1, end_row=1, start_column=3, end_column=7)
            wdt.merge_cells(start_row=1, end_row=1, start_column=8, end_column=12)
            wdt.merge_cells(start_row=1, end_row=1, start_column=13, end_column=16)
    wb.save(file)


if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter('inductor_log/' + str(args.suite) + '/Inductor_' + args.suite + '_E2E_Test_Report.xlsx')
    generate_report(excel, args.precision)
    excel_postprocess(excel, args.precision)