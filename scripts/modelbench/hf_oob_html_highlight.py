import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate HF testcase report")
parser.add_argument('-i', '--input', type=str, help='input csv file')
parser.add_argument('-o', '--output', type=str, help='output html file')
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

df_style = df.style.apply(highlight_greaterthan, threshold=1.05, column=['Compile ratio new/old'], axis=1).\
    apply(highlight_lessthan, threshold=0.95, column=['Compile ratio new/old'], axis=1).\
    apply(highlight_isnan, column=['Compile ratio new/old'], axis=1).\
    hide(axis="index")
df_style.to_html(args.output)
