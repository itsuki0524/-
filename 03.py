# 文から単語を抜き出して文字数をカウント
str = input("文を入力: ")
str_splits = str.split()
str_len = [len(i) for i in str_splits]
print(str_len)