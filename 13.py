# 二つのファイルを結合させるやつ
import pandas as pd
import os

col1 = pd.read_table('col1.txt')
col2 = pd.read_table('col2.txt')

merged_files = pd.concat([col1, col2], axis=1)

f_name = 'merged_files.txt'

with open(f_name, 'w') as f:
    f.write('')

merged_files.to_csv(f_name, sep='\t', index=False)