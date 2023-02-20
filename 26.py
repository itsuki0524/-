#強調マークアップを除去するやつ
import json 
import re

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
    for line in f:
        def remove_markup(text):

            pattern = r'\'{2,5}'
            text = re.sub(pattern, '', text)

        return text

        result_rm = {k: remove_markup(v) for k, v in result.items()}
        for k, v in result_rm.items():
            print(k + ': ' + v)