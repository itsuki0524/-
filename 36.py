#出現頻度が高い10語とその出現頻度をグラフで表示するやつ
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
        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

keys = [a[0] for a in ans[0:10]]
values = [a[1] for a in ans[0:10]]
plt.figure(figsize=(8, 4))
plt.bar(keys, values)
plt.show()