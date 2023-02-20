# 各行の3コラム目の数値の逆順にするやつ
import os
import pandas as pd
import sys

df = pd.read_csv(open(sys.argv[1]), sep='\t')

df.sort_values(by='number', ascending=False, inplace=True)