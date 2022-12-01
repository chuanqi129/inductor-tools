import argparse
parser = argparse.ArgumentParser(description='Inductor log parser')
parser.add_argument('--log-file', type=str, default="inductor.log", help='log file')
parser.add_argument('--compare-file', type=str, default=None, help='log file to compare')
args = parser.parse_args()

all_models = []
def parse_log(file):
    result = []
    with open(file, 'r') as reader:
        contents = reader.readlines()
        model = ""
        for line in contents:
            if "Time cost" in line:
                model = line.split(" Time cost")[0].split(" ")[-1].strip()
            elif line.startswith("eager: "):
                result.append(model+", "+ line)
            elif "cpu  eval" in line:
                m = line.split("cpu  eval")[-1].strip().split(" ")[0].strip()
                if m not in all_models:
                    all_models.append(m)
    return result

def str_to_dict(contents):
    res_dict = {}
    for line in contents:
        model = line.split(",")[0]
        eager = float(line.split(",")[1].strip().split(":")[-1])
        inductor = float(line.split(",")[2].strip().split(":")[-1])
        res_dict[model] = [eager, inductor]
    return res_dict

if args.compare_file is None:
    result = parse_log(args.log_file)
    print(result)
else:
    old_res = parse_log(args.compare_file)
    new_res = parse_log(args.log_file)
    old_res_dict = str_to_dict(old_res)
    new_res_dict = str_to_dict(new_res)
    results = ["Model, Eager(old), Inductor(old), Eager(new), Inductor(new), Eager Ratio(new/old), Inductor Ratio(new/old)\n"]
    for key in all_models:
        line = key+", "
        if key in old_res_dict:
            for item in old_res_dict[key]:
                line += str(item) +", "
        else:
            line += "NA, NA, "
        if key in new_res_dict:
            for item in new_res_dict[key]:
                line += str(item) +", "
        else:
            line += "NA, NA, "
        if key in old_res_dict and key in new_res_dict:
            line+=str(new_res_dict[key][0]/old_res_dict[key][0]) + ", "
            line+=str(new_res_dict[key][1]/old_res_dict[key][1])
        else:
            line += "NA, NA, "
        line += "\n"
        results.append(line)
    with open("results.csv", 'w') as writer:
        for line in results:
            writer.write(line)

