# 各行の1列目の文字列の出現頻度を求めるやつ
import os
import pandas as pd
import sys

df = pd.read_csv(open(sys.argv[1]), sep='\t')

print(df['name'].value_counts())