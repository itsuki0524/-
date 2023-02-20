# ファイルを読み込んで行数をカウントするやつ
import sys
import pandas as pd

df = pd.read_table(open(sys.argv[1]))

print(len(df))