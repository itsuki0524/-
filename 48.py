#名詞から根へのパスを抽出するやつ
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
for chunk in sentence.chunks:
    if '名詞' in [morph.pos for morph in chunk.morphs]:
        path = [''.join(morph.surface for morph in chunk.morphs if morph.pos != '記号')]
        while chunk.dst != -1:
            path.append(''.join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos != '記号'))
            chunk = sentence.chunks[chunk.dst]
        print(' -> '.join(path))