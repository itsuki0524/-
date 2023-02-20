# 任意の番号の単語を取り出して一部を出力するやつ
str = input("文を入力: ")
str_splits = str.split()
judge_num = [1, 5, 6, 7, 8, 9, 15, 16, 19]  # 取り出す単語の番号
associate_num = {}

for i, str_word in enumerate(str_splits):
    if i + 1 in judge_num:
        associate_num[str_word[:1]] = i + 1  # 真なら一文字
    else:
        associate_num[str_word[:2]] = i + 1  # 偽なら二文字

print(associate_num)