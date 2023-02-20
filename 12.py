# 一1列目と2列目を抜き出して別ファイルに保存するやつ
import sys
import os
import pandas as pd

df = pd.read_csv(open(sys.argv[1]), header=None, sep='\t', names=['name', 'sex', 'number', 'year'])

col1 = df.name
col2 = df.sex


f_name1 = 'col1.txt'
f_name2 = 'col2.txt'

with open(f_name1, 'w') as f1:
    f1.write('')

col1.to_csv(f_name1, index=False)

with open(f_name2, 'w') as f2:
    f2.write('')

col2.to_csv(f_name2, index=False)