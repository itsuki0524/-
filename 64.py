#ベクトルと類似度の高いものとその類似度を調べるやつ
import sys

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('./questions-words.txt', 'r') as f1, open('./questions-words-add.txt', 'w') as f2:
    for line in f1:
        line = line.split()
        if line[0] == ':':
            category = line[1]
        else:
            word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            f2.write(' '.join([category] + line + [word, str(cos) + '\n']))

    lines = f2.read.lines()

print("".join(lines[:int(10)]))