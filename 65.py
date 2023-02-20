#意味的アナロジーと文法的アナロジーの正解率を測定するやつ
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('./questions-words.txt', 'r') as f1, open('./questions-words-add.txt', 'w') as f2:
    for line in f1:
        line = line.split()
        if line[0] == ':':
            category = line[1]
        else:
            word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            f2.write(' '.join([category] + line + [word, str(cos) + '\n']))

    sem_cnt = 0
    sem_cor = 0
    syn_cnt = 0
    syn_cor = 0
    for line in f2:
        line = line.split()
        if not line[0].startswith('gram'):
            sem_cnt += 1
            if line[4] == line[5]:
                sem_cor += 1
            else:
                syn_cnt += 1
                if line[4] == line[5]:
                    syn_cor += 1

print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}') 