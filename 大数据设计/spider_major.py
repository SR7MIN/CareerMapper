import json
import jsonpath
import requests
import os
from bs4 import BeautifulSoup


def get_info(id):
    
    addr = 'https://static-data.gaokao.cn/www/2.0/special/{0}/pc_special_detail.json'.format(id)
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    response = requests.get(addr, headers=headers)
    status_code = response.status_code
    if status_code != 200:
        return
    html_str = response.content.decode()
    jsonobj = json.loads(html_str)
    name = jsonobj['data']['name']
    file = open(os.path.join('./major', name + ".txt"), "w")
    message = jsonobj['data']['jobdetail']
    if message.get('1') is not None:
        file.write(','.join(jsonpath.jsonpath(jsonobj, '$.data.jobdetail.1[*].name')))
        file.write('\n')
        file.write(','.join(jsonpath.jsonpath(jsonobj, '$.data.jobdetail.1[*].rate')))
        file.write('\n')
    else:
        file.write('无')
    file.close()


for i in range(300, 20001):
    get_info(i)

