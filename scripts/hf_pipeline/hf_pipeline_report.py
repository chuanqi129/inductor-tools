import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Generate HF testcase report")
parser.add_argument('-t', '--target', type=str, help='target log file folder')
parser.add_argument('-r', '--reference', type=str, help='reference log file folder')
parser.add_argument('-f', '--files', type=str, default='cpu_output_OOB.log,cpu_output_compile.log', help='files to analyze')
args = parser.parse_args()

def analyze_log(file_dir, file_name):
    file_path = '{0}/hf_pipeline_log/{1}'.format(file_dir, file_name)
    file = open(file_path, 'r')

    lines = file.readlines()
    write_line = ""
    task_name = ""
    output_file_name = "{0}_{1}_output.csv".format(file_dir, file_name.split('.')[0])
    write_file = open(output_file_name, "w")
    latency_name = "{0} Eager latency".format(file_dir)
    if "compile" in file_name:
        latency_name = "{0} Compile latency".format(file_dir)
    title = "Task,Model,{0}\n".format(latency_name)
    write_file.write(title)
    for line in lines:
        if line.startswith('test '):
            # if model failed and has no avg time, the data len == 2.
            if len(write_line.split(',')) == 3:
                write_line += ','
                write_file.write(write_line)
                write_file.write('\n')
            write_line = ""
            task_name = line.split('test ')[1].strip()
        elif line.startswith('INFO:root:args'):
            model_name_list = line.split('model_id=')
            model_name = model_name_list[1].split("'")[1]
            precision = line.split('model_dtype=')[1].split("'")[1]
            write_line = "{0},{1},{2}".format(task_name, model_name, precision)
        elif line.startswith('INFO:root:pipeline'):
            time_list = [] 
            if "[ms]:" in line:
                time_list = line.split('average time [ms]: ')
            else:
                time_list = line.split('average time [ms] ')
            avg_time = time_list[1].split()[0].strip(',')
            write_line = write_line + "," + avg_time
            write_file.write(write_line)
            write_file.write('\n')
    write_file.close()
    file.close()

def merge_tables(file_dir, files):
    output_file_name_1 = "{0}_{1}_output.csv".format(file_dir, files[0].split('.')[0])
    output_file_name_2 = "{0}_{1}_output.csv".format(file_dir, files[1].split('.')[0])
    data_df_1 = pd.read_csv(output_file_name_1)    
    data_df_2 = pd.read_csv(output_file_name_2)    
    summary_df = pd.merge(data_df_1, data_df_2, how='outer')
    summary_df['{0} Eager/Compile ratio'.format(file_dir)] = \
        summary_df['{0} Eager latency'.format(file_dir)] / summary_df["{0} Compile latency".format(file_dir)]
    return summary_df

def generate_summary(file_dir):
    files = args.files.split(',')
    for file_name in files:
       analyze_log(file_dir, file_name)

    if len(files) == 2:
         return merge_tables(file_dir, files)
    else:
        return None

def merge_refer_tables(target_df, refer_df):
    summary_df = pd.merge(target_df, refer_df, how='outer')
    summary_df['Eager ratio old/new'] = refer_df["{0} Eager latency".format(args.reference)]/target_df["{0} Eager latency".format(args.target)]
    summary_df['Compile ratio old/new'] = refer_df["{0} Compile latency".format(args.reference)]/target_df["{0} Compile latency".format(args.target)]
    return summary_df

def get_sw_df(file_dir):
    sw_df = pd.read_csv("{0}/hf_pipeline_log/version.csv".format(file_dir))
    sw_df = sw_df.rename(columns={
        'branch':'{0}_branch'.format(file_dir),
        'commit':'{0}_commit'.format(file_dir)})
    return sw_df

def main():
    # Step 1: Analyze log and extract usefuly data save to .csv
    target_df = generate_summary(args.target)
    if args.reference is not None:
        refer_df = generate_summary(args.reference)
        target_df = merge_refer_tables(target_df, refer_df)    
    if target_df is not None:
        target_df.to_csv('summary.csv', index=False)
    else:
        print("No target dataframe found")
    
    # TODO: Add SW & HW info
    target_sw_df = get_sw_df(args.target)
    if args.reference is not None:
        refer_sw_df = get_sw_df(args.reference)
        target_sw_df = pd.merge(target_sw_df, refer_sw_df)
    target_sw_df.to_csv('version_summary.csv', index=False)

if __name__ == "__main__":
    main()
