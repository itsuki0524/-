# 二つの単語の要素の集合を求めるやつ
def n_gram(n, str_list):
  return list(zip(*[str_list[i:] for i in range(n)]))

str1 = input("一つ目の単語を入力: ")
str2 = input("二つ目の単語を入力: ")
X = set(n_gram(2, str1))
Y = set(n_gram(2, str2))
str_union = X | Y
str_intersec = X & Y
str_diff = X - Y

print('X:', X)
print('Y:', Y)
print('和集合:', str_union)
print('積集合:', str_intersec)
print('差集合:', str_diff)
print('Xにseが含まれるか:', {('s', 'e')} <= X)
print('Yにseが含まれるか:', {('s', 'e')} <= Y)