#k-meansクラスタリングを実行するやつ
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

countries = set()
with open('./questions-words.txt') as f:
    for line in f:
        line = line.split()
        if line[0] in ['capital-common-countries', 'capital-world']:
            countries.add(line[2])
        elif line[0] in ['currency', 'gram6-nationality-adjective']:
            countries.add(line[1])
countries = list(countries)

countries_vec = [model[country] for country in countries]

from sklearn.cluster import KMeans

#k-meansクラスタリング
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)
for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print('cluster', i)
        print(', '.join([countries[k] for k in cluster]))