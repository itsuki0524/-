#「猫」と共起頻度の高い上位10語をグラフで表示するやつ
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
            if '猫' in [morph['surface'] for morph in sentence]:
                for morph in sentence:
                    if morph['pos'] != '記号':
                        ans[morph['base']] += 1
        del ans['猫']
        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

keys = [a[0] for a in ans[0:10]]
values = [a[1] for a in ans[0:10]]
plt.figure(figsize=(8, 4))
plt.bar(keys, values)
plt.show()