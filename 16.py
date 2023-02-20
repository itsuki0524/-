# ファイルをN分割するやつ
import os
import pandas as pd
import sys

n = input('分割したい数: ')

df = pd.read_csv(open(sys.argv[1]), sep='\t')

def split_file(N):
  tmp = df.reset_index(drop=False)
  df_cut = pd.qcut(tmp.index, N, labels=[i for i in range(N)])
  df_cut = pd.concat([df, pd.Series(df_cut, name='sp')], axis=1)

  return df_cut

df_cut = split_file(n)
print(df_cut['sp'].value_counts())