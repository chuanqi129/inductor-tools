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
    res_dict = {}
    for model in df.name:
        model_df = df.loc[df["name"] == model]
        batch_size = bs if model_df.empty else model_df['batch_size'].values[0]
        speedup = model_df["speedup"].values[0]
        inductor_latency = model_df["abs_latency"].values[0] / 1000
        eager_latency = inductor_latency * speedup
        inductor_th = batch_size / inductor_latency
        eager_th = batch_size / eager_latency
        res_dict[model] = [batch_size, eager_th, inductor_th, speedup, inductor_latency]
        if model not in all_models:
            all_models.append(model)
    return res_dict

all_results = []
title = "Model, BS, Eager, Inductor, Speedup, Inductor_latency\n"
for file in get_log_files(args.tune_dir):
    all_results.append(parse_log(file))

final_res = [title]

for model in all_models:
    for single_res in all_results:
        line = model
        if model in single_res:
            for item in single_res[model]:
                line += ", " + str(item)
        else:
            line += ", NA, NA, NA, NA, NA"
        line += "\n"
        final_res.append(line)
with open(args.tune_dir + "/"+ args.suite +"_tuning.csv", 'w') as writer:
    for line in final_res:
        writer.write(line)
