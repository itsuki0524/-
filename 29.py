#国旗画像のURLを取得するやつ
import json 
import re
import requests

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
    for line in f:
        def get_url(text):
            url_file = text['国旗画像'].replace(' ', '_')
            url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
            data = requests.get(url)
            return re.search(r'"url":"(.+?)"', data.text).group(1)

print(get_url(result))