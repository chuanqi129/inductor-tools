import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Generate HF testcase report")
parser.add_argument('-t', '--target', type=str, help='target log file folder')
parser.add_argument('-r', '--reference', type=str, help='reference log file folder')
parser.add_argument('-f', '--files', type=str, default='cpu_output_OOB.log,cpu_output_compile.log', help='files to analyze')
args = parser.parse_args()

def analyze_log(file_name):
    file_path = '{0}/hf_oob_log/{1}'.format(args.target, file_name)
    file = open(file_path, 'r')

    lines = file.readlines()
    write_line = ""
    task_name = ""
    output_file_name = "{0}_output.csv".format(file_name.split('.')[0])
    write_file = open(output_file_name, "w")
    latency_name = "Eager latency"
    if "compile" in file_name:
        latency_name = "Compile latency"
    title = "Task,Model,{0}\n".format(latency_name)
    write_file.write(title)
    for line in lines:
        if line.startswith('test '):
            # if model failed and has no avg time, the data len == 2.
            if len(write_line.split(',')) == 2:
                write_line += ','
                write_file.write(write_line)
                write_file.write('\n')
            write_line = ""
            task_name = line.split('test ')[1].strip()
        elif line.startswith('INFO:root:args'):
            model_name_list = line.split('model_id=')
            model_name = model_name_list[1].split("'")[1]
            write_line = task_name + "," + model_name
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

def merge_tables(files):
    output_file_name_1 = "{0}_output.csv".format(files[0].split('.')[0])
    output_file_name_2 = "{0}_output.csv".format(files[1].split('.')[0])
    data_df_1 = pd.read_csv(output_file_name_1)    
    data_df_2 = pd.read_csv(output_file_name_2)    
    summary_df = pd.merge(data_df_1, data_df_2, how='outer')
    summary_df['Eager/Compile ratio'] = summary_df['Eager latency'] / summary_df['Compile latency']
    summary_df.to_csv('summary.csv', index=False)

def main():
    files = args.files.split(',')
    #for file_name in files:
    #    analyze_log(file_name)
    if len(files) == 2:
        merge_tables(files)

if __name__ == "__main__":
    main()
