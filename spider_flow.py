import json
import jsonpath
import requests
from bs4 import BeautifulSoup

def get_computer_science_info():
    """格式化爬取计算机科学与技术专业就业数据
    
    写入记事本的数据结构如下：（在读取数据时可直接参考这个结构来遍历读取）
    第一行：就业领域划分1，相邻两个就业领域以','分割，在读取数据时若不方便可在源码中将其改为空格
    第二行：对应领域划分1的所有占比，每一个数据的迭代下标与每一个领域一一对应
    第三行-第八行：就业领域划分2，每一行为一个就业领域的划分，即一个大就业领域中的各个分支综合
    第九行：对应第三到八行的每个就业领域的划分之和，以逗号相隔
    第十行：三个不同年份
    第十一行：对应第十行中三个年份的专业就业率"""
    addr='https://static-data.gaokao.cn/www/2.0/special/172/pc_special_detail.json'
    headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    response=requests.get(addr,headers=headers)
    html_str=response.content.decode()
    jsonobj=json.loads(html_str)
    file=open("computer science.txt","w")
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.1[*].name')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.1[*].rate')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.2[*].area')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.2[*].rate')))
    file.write('\n')
    file.write('\n'.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.3[*].detail_job')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.3[*].detail_pos')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobdetail.3[*].rate')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobrate[*].year')))
    file.write('\n')
    file.write(','.join(jsonpath.jsonpath(jsonobj,'$.data.jobrate[*].rate')))
    file.write('\n')
    file.close()



#做了好多无用功QAQ
