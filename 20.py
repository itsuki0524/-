# 記事のJSONファイルを読み込むやつ
import json 

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
  for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
      text_article = line['text']
      break

print(text_article)