#単語の出現頻度順位をグラフで表示するやつ
import matplotlib.pyplot as plt
import japanize_matplotlib
import math

from collections import defaultdict 

filename = './neko.txt.mecab'

sentences = []
morphs = []
with open(filename, mode='r') as f:
    for line in f:
        ans = defaultdict(int)
        for sentence in sentences:
            for morph in sentence:
                if morph['pos'] != '記号':
                    ans[morph['base']] += 1
        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

ranks = [r + 1 for r in range(len(ans))]
values = [a[1] for a in ans]
plt.figure(figsize=(8, 4))
plt.scatter(ranks, values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.show()