#単語の出現頻度のヒストグラムを表示するやつ
import matplotlib.pyplot as plt
import japanize_matplotlib

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
        ans = ans.values()

plt.figure(figsize=(8, 4))
plt.hist(ans, bins=100)
plt.xlabel('出現頻度')
plt.ylabel('単語の種類数')
plt.show()