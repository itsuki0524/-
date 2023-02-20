# 文から要素を抽出して組を作るやつ
def n_gram(n, str_list):
  return list(zip(*[str_list[i:] for i in range(n)]))

str = input("シーケンスを入力: ")
word_bi_gram = n_gram(2, str.split())
char_bi_gram = n_gram(2, str)

print('単語bi-gram:', word_bi_gram)
print('文字bi-gram:', char_bi_gram)