import sys
import os
import pandas as pd

df = pd.read_table(open(sys.argv[1]), sep='\t', names=['name', 'sex', 'number', 'year'])

col1 = df.name
col2 = df.sex

print(col1)