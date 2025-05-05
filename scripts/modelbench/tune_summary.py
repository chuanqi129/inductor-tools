
import argparse
import os

parser = argparse.ArgumentParser(description="information for modifing .py")
parser.add_argument("--file_path", required=True, help="input file path, e.g. /path/test.log")
args = parser.parse_args()
files = os.listdir(args.file_path)
save_path="./summary_tune.log"
with open(save_path, 'w') as save_file:
    for file in files:
        print(file)
        filename=file.rsplit(".", 1)[0]
        # file="test_tp_style.log"
        if file.endswith("log"):
            save_file.write(str(filename) + ",")
            file_path = args.file_path+file
            with open(file_path, 'r') as read_file:
                lines = read_file.readlines() 
                for i, line in enumerate(lines):
                    if "avg " in line:
                        print(line)
                        save_file.write(line)
            read_file.close()
save_file.close()

