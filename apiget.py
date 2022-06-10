import requests
import json

ak='rHLnzMg7mKPFNWYv3QX9FXkv'
sk='gjVKdfrd2Y7IN4OzcoTzFvXeMEOG9Zv0'
# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=rHLnzMg7mKPFNWYv3QX9FXkv&client_secret=gjVKdfrd2Y7IN4OzcoTzFvXeMEOG9Zv0'
response = requests.get(host)
if response:
    print(response.json())
n=json.loads(response.text)
print(n["access_token"])