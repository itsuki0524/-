# 区切りをTABからスペースに変えるやつ
import sys
import os

with open(sys.argv[1], 'r') as f1:
    df = f1.read()
    replace_f = df.replace('\t', ' ')



f_name = 'replace_' + sys.argv[1]

with open(f_name, 'w') as f2:
    f2.write(replace_f)