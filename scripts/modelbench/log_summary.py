import argparse
import os
parser = argparse.ArgumentParser(description='Inductor log parser')
parser.add_argument('--logs', type=str, default="inductor.log", help='log file')
args = parser.parse_args()
files = os.listdir(args.logs)
with open('summary.log', 'w') as summary:
    summary.write('model,dtype,backend,input_length,max_new_tokens,groupsize,qconfig,total,first token,next token\n')
    for file in files:
        file_path = os.path.join(args.logs, file)
        filename = os.path.splitext(file)[0]
        filename = filename.replace("aot_q", "aotq")
        model,dtype,compile,input_length,max_new_tokens,gps,qconfig = filename.split("_")
        flag = 0
        with open(file_path, 'r') as file:
            for line in file:
                if 'Total throughput' in line:
                    flag = 1
                    total_throughput = line.split(' ')[8]
                if 'First token throughput' in line:
                    first_throughput = line.split(' ')[3]
                if 'Next token throughput' in line:
                    next_throughput = line.split(' ')[4]
        if flag == 1:
            summary.write(model+','+dtype+','+compile+','+input_length+','+max_new_tokens+','+gps+','+qconfig+','+total_throughput+','+first_throughput+','+next_throughput+'\n')
summary.close()