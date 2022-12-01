import argparse

parser = argparse.ArgumentParser(description='PyTorch models inference')
parser.add_argument('-f', '--file', type=str, help='log file')
parser.add_argument('-o', '--output', type=str, help='output file for parser results')
args = parser.parse_args()
success_ops = {}
error_ops = {}
skipped_ops = {}
with open(args.file) as reader:
    lines = reader.readlines()
    op = ""
    for line in lines:
        if line.strip() == "":
            continue
        if line.startswith("Running"):
            op = line.split(" ")[1].strip()
        # Skipping aten.convolution_backward.default, no inductor impl
        elif line.startswith("Skipping"):
            op = line.split(",")[0].split(" ")[1].strip()
            reason = line.split(",")[1].strip()
            if op not in skipped_ops.keys():
                skipped_ops[op] = reason
                # print(op+", " + reason)
        # error aten.stack.default
        elif line.startswith("error"):
            op = line.split(" ")[1].strip()
            if op not in error_ops.keys():
                error_ops[op] = "error"
                # print(op + ", error")
        if line.startswith("Inductor"):
            speedups = line.split("[")[-1].split("]")[0]
            if op not in success_ops.keys():
                success_ops[op] = speedups
                # print(op+", "+speedups)

if (args.output):
    with open(args.output, 'w') as writer:
        writer.write("op_name, speedup_0.2, speedup_0.5, speedup_0.8\n")
        for op in sorted(success_ops):
            writer.write(op + ", " + success_ops[op] + "\n")
        for op in sorted(skipped_ops):
            writer.write(op + ", " + skipped_ops[op] + "\n")
        for op in sorted(error_ops):
            writer.write(op + ", " + error_ops[op] + "\n")
else:
    print("op_name, speedup_0.2, speedup_0.5, speedup_0.8")
    for op in sorted(success_ops):
        print(op + ", " + success_ops[op])
    for op in sorted(skipped_ops):
        print(op + ", " + skipped_ops[op])
    for op in sorted(error_ops):
        print(op + ", " + error_ops[op])
