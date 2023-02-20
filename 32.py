#動詞の原形を抽出するやつ
filename = './neko.txt.mecab'

sentences = []
morphs = []
with open(filename, mode='r') as f:
    for line in f:
        ans = set()
        for sentence in sentences:
            for morph in sentence:
                if morph['pos'] == '動詞':
                    ans.add(morph['base'])