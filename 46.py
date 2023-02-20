#動詞の格フレーム情報を抽出するやつ
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

with open('./case_frame.txt', 'w') as f:
    for sentence in sentences:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    cases = []
                    modi_chunks = []

                    for src in chunk.srcs:
                        case = [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == '助詞']
                        if len(case) > 0:
                            cases = cases + case
                            modi_chunks.append(''.join(morph.surface for morph in sentence.chunks[src].morphs if morph.pos != '記号'))
                    
                    if len(cases) > 0:
                        cases = sorted(list(set(cases)))
                        line = '{}\t{}\t{}'.format(morph.base, ' '.join(cases), ' '.join(modi_chunks))
                        print(line, file=f)
                    
                    break