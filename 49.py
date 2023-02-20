#名詞間の係り受けパスを抽出するやつ
import re

from itertools import combinations

class Morph:
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr = attr.split(',')
        self.surface = surface
        self.base = attr[6]
        self.pos = attr[0]
        self.pos1 = attr[1]

class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []

class Sentence():
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:
                self.chunks[chunk.dst].srcs.append(i)

filename = './ai.ja.txt.parsed'

sentences = []
morphs = []
with open(filename, mode='r') as f:
    for line in f:
        if line[0] == '*':
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))
                morphs = []
            dst = int(line.split(' ')[2].rstrip('D'))
        elif line != 'EOS\n':
            morphs.append(Morph(line))
        else:
            chunks.append(Chunk(morphs, dst))
            sentences.append(Sentence(chunks))
            morphs = []
            chunks = []
            dst = None

sentence = sentences[2]
nouns = []

for i, chunk in enumerate(sentence.chunks):
    if '名詞' in [morph.pos for morph in chunk.morphs]:
        nouns.append(i)

for i, j in combinations(nouns, 2):
    path_i = []
    path_j = []

    while i != j:
        if i < j:
            path_i.append(i)
            i = sentence.chunks[i].dst
        else:
            path_j.append(j)
            j = sentence.chunks[j].dst
  
    if len(path_j) == 0:  # 1つ目のケース
        chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in sentence.chunks[path_i[0]].morphs])
        chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in sentence.chunks[i].morphs])
        chunk_X = re.sub('X+', 'X', chunk_X)
        chunk_Y = re.sub('Y+', 'Y', chunk_Y)
        path_XtoY = [chunk_X] + [''.join(morph.surface for morph in sentence.chunks[n].morphs) for n in path_i[1:]] + [chunk_Y]
        print(' -> '.join(path_XtoY))
    else:  # 2つ目のケース
        chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in sentence.chunks[path_i[0]].morphs])
        chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in sentence.chunks[path_j[0]].morphs])
        chunk_k = ''.join([morph.surface for morph in sentence.chunks[i].morphs])
        chunk_X = re.sub('X+', 'X', chunk_X)
        chunk_Y = re.sub('Y+', 'Y', chunk_Y)
        path_X = [chunk_X] + [''.join(morph.surface for morph in sentence.chunks[n].morphs) for n in path_i[1:]]
        path_Y = [chunk_Y] + [''.join(morph.surface for morph in sentence.chunks[n].morphs) for n in path_j[1:]]
        print(' | '.join([' -> '.join(path_X), ' -> '.join(path_Y), chunk_k]))