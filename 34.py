#名詞の連接を抽出するやつ
filename = './neko.txt.mecab'

sentences = []
morphs = []
with open(filename, mode='r') as f:
    for line in f:
        ans = set()
        for sentence in sentences:
            nouns = ''
            num = 0
            for morph in sentence:
                if morph['pos'] == '名詞':
                    nouns = ''.join([nouns, morph['surface']])
                    num += 1
                elif num >= 2:
                    ans.add(nouns)
                    nouns = ''
                    num = 0
                else:
                    nouns = ''
                    num = 0
        if num >= 2: 
            ans.add(nouns)