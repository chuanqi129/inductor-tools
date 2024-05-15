import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate HF testcase report")
parser.add_argument('-i', '--input', type=str, help='input csv file')
parser.add_argument('-o', '--output', type=str, help='output html file')
parser.add_argument('-t', '--target', type=str, help='target build')
parser.add_argument('-r', '--refer', type=str, help='refer build')
parser.add_argument('--threshold', type=float, default=0.1, help="threshold for comparing new and old ratio")
args = parser.parse_args()

def highlight_greaterthan(data, threshold, column):
    is_max = pd.Series(data=False, index=data.index)
    is_max[column] = data.loc[column] > threshold
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]

def highlight_lessthan(data, threshold, column):
    is_max = pd.Series(data=False, index=data.index)
    is_max[column] = data.loc[column] < threshold
    return ['background-color: #ACFF33' if is_max.any() else '' for v in is_max]

def highlight_isnan(data, column):
    is_nan = pd.Series(data=False, index=data.index)
    is_nan[column] = data.loc[column].isna()
    return ['background-color: #FF7A33' if is_nan.any() else '' for v in is_nan]

df = pd.read_csv(args.input)

if args.refer == "0":
    ratio_subset = ['{0} Torch.compile vs. eager Speedup'.format(args.target)]
    df_style = df.style.hide(axis="index").\
        highlight_between(left=0,right=1,subset=ratio_subset).\
        highlight_null(subset=ratio_subset)
else:
    ratio_subset = ['{0} Torch.compile vs. eager Speedup'.format(args.target), 'refer Torch.compile vs. eager Speedup']
    old_new_subset = ['Eager ratio old/new', 'Compile ratio old/new']
    df_style = df.style.hide(axis="index").\
        highlight_between(left=0,right=1,subset=ratio_subset).\
        highlight_null(subset=ratio_subset).\
        highlight_between(left=0,right=1-args.threshold,subset=old_new_subset,props='color:black;background-color:#FF7A33').\
        highlight_between(left=1+args.threshold,right=float('inf'),subset=old_new_subset,props='color:black;background-color:#ACFF33').\
        highlight_null(subset=old_new_subset)
df_style.to_html(args.output)
