#メディアファイルを抽出するやつ
import json 
import re

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
    for line in f:
        pattern = r'\[\[ファイル:(.+?)\|'
        result = '\n'.join(re.findall(pattern, text_uk))

print(result)