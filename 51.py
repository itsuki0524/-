#特徴量を抽出するやつ
import string
import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

df = pd.concat([train, valid, test], axis=0)
df.reset_index(drop=True, inplace=True)

df['TITLE'] = df['TITLE'].map(lambda x: preprocessing(x))

train_valid = df[:len(train) + len(valid)]
test = df[len(train) + len(valid):]


vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE'])
X_test = vec_tfidf.transform(test['TITLE'])

X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names())
X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names())

X_train = X_train_valid[:len(train)]
X_valid = X_train_valid[len(train):]

X_train.to_csv('./X_train.txt', sep='\t', index=False)
X_valid.to_csv('./X_valid.txt', sep='\t', index=False)
X_test.to_csv('./X_test.txt', sep='\t', index=False)