# 1列目の文字列の種類を求めるやつ
import os
import pandas as pd
import sys

df = pd.read_csv(open(sys.argv[1]), sep='\t')

print(len(df.drop_duplicates(subset='name')))