#単語の出現頻度を出力するやつ
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

for w in ans[:10]:
  print(w)