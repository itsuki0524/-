# 指定した末尾からの行数を出力するやつ
import os
import pandas as pd
import sys

n = input('出力したい末尾の行数: ')

df = pd.read_csv(open(sys.argv[1]), sep='\t')

print(df.tail(n))