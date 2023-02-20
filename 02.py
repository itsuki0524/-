# 文字列を交互に足し算して出力するやつ
str_1 = input("一つ目の文字列を入力: ")
str_2 = input("二つ目の文字列を入力: ")
str = ''.join([i + j for i, j in zip(str_1, str_2)])
print(str)