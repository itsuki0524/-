#ベクトルと類似度の高いものとその類似度を調べるやつ
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

vec = model['Spain'] - model['madrid'] + model['Athens'] 
model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)