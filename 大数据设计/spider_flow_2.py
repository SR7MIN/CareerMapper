import csv
from bs4 import BeautifulSoup
import requests
import json



def get_people_num():
    cater_ids = []
    ids = []
    name = '专业人数.csv'
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    for i in range(1, 13):
        url_1 = 'https://gaokao.chsi.com.cn/zyk/zybk/xkCategory/1050{:02}'.format(i)
        response = requests.get(url_1, headers=headers)
        str = response.content.decode()
        jsonobj = json.loads(str)
        for item in jsonobj['msg']:
            cater_ids.append(item['key'])

    for id in cater_ids:
        url_2 = 'https://gaokao.chsi.com.cn/zyk/zybk/specialityesByCategory/' + id
        response = requests.get(url_2, headers=headers)
        str = response.content.decode()
        jsonobj = json.loads(str)
        for item in jsonobj['msg']:
            ids.append(item['specId'])

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        for id in ids:
            url_3 = 'https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/' + id
            response = requests.get(url_3, headers=headers)
            str = response.content.decode()
            jsonobj = json.loads(str)
            if jsonobj['msg']['xlcc'] == '本科（普通教育）':
                writer.writerow([jsonobj['msg']['zymc'], jsonobj['msg']['xsgm']])


get_people_num()
# str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityesByCategory/10500102')
# jsonobj=json.loads(str)
# file=open("各专业人数.txt","w")
# file_write(file,jsonobj)


