#テンプレートを抽出するやつ
import json 
import re

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
    for line in f:
        pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}'
        template = re.findall(pattern, text_uk, re.MULTILINE + re.DOTALL)

        pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
        result = dict(re.findall(pattern, template[0], re.MULTILINE + re.DOTALL))
        for k, v in result.items():
            d_object = (k + ': ' + v)


print(template)
