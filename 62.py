#その単語の類似度の高いものとその類似度を調べるやつ
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

model.most_similar('United_States', topn=10)