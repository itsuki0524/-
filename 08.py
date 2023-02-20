# 暗号化するやつ
def cipher(str):
  rep = [chr(219 - ord(n)) if n.islower() else n for n in str]
  return ''.join(rep)

str = input('文章を入力: ')
str = cipher(str)
print('暗号化:', str)
str = cipher(str)
print('復号化:', str)