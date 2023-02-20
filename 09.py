# 入力された文の一定以上の文字数の単語を並び変えるやつ
import random

def shuffle(str):
  result = []
  for word in str.split():
    if len(word) > 4:  # 長さが4超であればシャッフル
      word = word[:1] + ''.join(random.sample(word[1:-1], len(word) - 2)) + word[-1:]
    result.append(word)

  return ' '.join(result)

str = input('文章を入力: ')
str = shuffle(str)

print(str)