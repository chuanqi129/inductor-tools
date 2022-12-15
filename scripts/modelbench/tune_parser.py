import argparse
import os
import pandas as pd
import re
parser = argparse.ArgumentParser(description='Inductor BS tuning log parser')
parser.add_argument('--suite', type=str, default="", help='model suite')
parser.add_argument('--tune-dir', type=str, default="", help='tuning log dir')
args = parser.parse_args()


def get_log_files(path):
    all_files = os.listdir(path)
    good_files = [
        path + file for file in all_files if file.startswith("multi_threads_model_bench_bs_")]
    return good_files


all_models = []


def parse_log(file):
    bs = re.findall("_bs_\d+_", file)[0].split("_")[2]
    date = re.findall("_\d+_\d+", file)[0]
    perf_csv_file = os.path.dirname(file) + "/multi_threads_cf_bs_" + bs + "_logs" + \
        date + "/inductor_" + args.suite + "_float32_inference_cpu_performance.csv"
    print(perf_csv_file)
    df = pd.read_csv(perf_csv_file)
    result = []
    with open(file, 'r') as reader:
        contents = reader.readlines()
        model = ""
        for line in contents:
            if "Time cost" in line:
                model = line.split(" Time cost")[0].split(" ")[-1].strip()
            elif line.startswith("eager: "):
                temp_df = df.loc[df["name"] == model]
                batch_size = bs if temp_df.empty else temp_df['batch_size'].values[0]
                result.append(model+", BS: " + str(batch_size) + ", " + line)
            elif "cpu  eval" in line:
                m = line.split("cpu  eval")[-1].strip().split(" ")[0].strip()
                if m not in all_models:
                    all_models.append(m)
    return result


def str_to_dict(contents):
    res_dict = {}
    for line in contents:
        model = line.split(",")[0]
        bs = int(line.split(",")[1].strip().split(":")[-1])
        eager = bs / float(line.split(",")[2].strip().split(":")[-1])
        inductor = bs / float(line.split(",")[3].strip().split(":")[-1])
        speedup = inductor / eager
        res_dict[model] = [bs, eager, inductor, speedup]
    return res_dict


all_results = []
title = "Model, BS, Eager, Inductor, Speedup\n"
for file in get_log_files(args.tune_dir):
    all_results.append(str_to_dict(parse_log(file)))

final_res = [title]

for model in all_models:
    for single_res in all_results:
        line = model
        if model in single_res:
            for item in single_res[model]:
                line += ", " + str(item)
        else:
            line += ", NA, NA, NA, NA"
        line += "\n"
        final_res.append(line)
with open(args.tune_dir + "/"+ args.suite +"_tuning.csv", 'w') as writer:
    for line in final_res:
        writer.write(line)
