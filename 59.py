#検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めるやつ
import string
import re
import optuna

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])

def objective_lg(trial):
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    C = trial.suggest_loguniform('C', 1e-4, 1e4)

    lg = LogisticRegression(random_state=123, 
                            max_iter=10000, 
                            penalty='elasticnet', 
                            solver='saga', 
                            l1_ratio=l1_ratio, 
                            C=C)
    lg.fit(X_train, train['CATEGORY'])

    valid_pred = score_lg(lg, X_valid)

    valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])    

    return valid_accuracy 

study = optuna.create_study(direction='maximize')
study.optimize(objective_lg, timeout=3600)

#最適なパラメータで再学習
l1_ratio = trial.params['l1_ratio']
C = trial.params['C']

lg = LogisticRegression(random_state=123, 
                        max_iter=10000, 
                        penalty='elasticnet', 
                        solver='saga', 
                        l1_ratio=l1_ratio, 
                        C=C)
lg.fit(X_train, train['CATEGORY'])


train_pred = score_lg(lg, X_train)
valid_pred = score_lg(lg, X_valid)
test_pred = score_lg(lg, X_test)

train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1]) 
valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1]) 
test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1]) 


print('Best trial:')
trial = study.best_trial
print('  Value: {:.3f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（検証データ）：{valid_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')