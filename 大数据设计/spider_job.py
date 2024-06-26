import json
import jsonpath
import requests

import time
import threading
import urllib3.contrib.pyopenssl
from threading import Lock


mutex = Lock()
    
urls=[]
urls.append("https://www.ncss.cn/student/jobs/jobslist/ajax/?jobType=&areaCode=&jobName=&monthPay=&industrySectors=&recruitType=&property=&categoryCode=&memberLevel=&offset=1&limit=10&keyUnits=&degreeCode=&sourcesName=&sourcesType=&_=1707310744313")
id=1
all_index = 17192
all_thread = 7
job_num = 0

num = int(all_index / all_thread)
for i in range(1, all_index):
    urls.append("https://www.ncss.cn/student/jobs/jobslist/ajax/?jobType=&areaCode=&jobName=&monthPay=&industrySectors=&recruitType=&property=&categoryCode=&memberLevel=&offset={0}&limit=10&keyUnits=&degreeCode=&sourcesName=&sourcesType=&_=1707310744313".format(i))    

file=open("current_jobs.txt","w",encoding='utf-8')    #file=open("current_jobs.txt","w")
#
#     特别注意
#
#这里的编码方式是后面加上的utf-8，一开始coder写的时候没有强制它utf-8编码，但是在后面随机选取3000条数据的时候出现了乱码问题！！！如果utf-8编码报错
#请去除该编码方式，在爬取后手动改变current_jobs的编码方式，从ANSI改为utf-8！！！
#
start=time.time()

class myThread(threading.Thread):
    def __init__(self,name,link_range):
        threading.Thread.__init__(self)
        self.name=name
        self.link_range=link_range
    def run(self):
        print("Starting"+self.name)
        crawl(self.name,self.link_range)
        print("Exiting"+self.name)

all_text_lines = []

def crawl(threadName,link_range):
    for i in range(link_range[0],link_range[1]+1):
        try:
            s = requests.session()      
            s.keep_alive =False
            urllib3.contrib.pyopenssl.inject_into_urllib3()
            r=requests.get(urls[i],timeout=1.5,headers={'Connection':'close'})
            requests.adapters.DEFAULT_RETRIES =5
            response_str=r.content.decode()
            jsonobj=json.loads(response_str)
            mutex.acquire()
            for j in range(len(jsonobj['data']['list'])):
                file.write(','.join(jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].jobName')))  #岗位名称
                file.write("|")
                file.write(','.join(jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].recName')))  #单位名称
                file.write('|')
                file.write(','.join(map(str,jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].highMonthPay'))))   #最高月薪
                file.write("|")
                file.write(','.join(map(str,jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].lowMonthPay'))))    #最低月薪
                file.write('|')
                file.write(','.join(map(str,jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].headCount'))))    #招聘人数
                file.write('|')
                file.write(','.join(jsonpath.jsonpath(jsonobj,f'$.data.list[{j}].degreeName')))   #要求学历
                file.write('\n')
            mutex.release()
            # print(threadName,r.status_code,urls[i])
        except Exception as e:
            pass
            # print(threadName,"Error:",e)
        global job_num
        job_num += 1
        print('\r爬取进度：{0}/{1}'.format(job_num, all_index), end="")

threads=[]
link_range_list = [(i*num, (i + 1)*num) for i in range(all_thread)]
# link_range_list = [(1,2456),(2456,4912),(4912,7367),(7367,9824),(9824,12279),(12279,14735),(14735,17191)]

for i in range(all_thread):
    thread=myThread("Thread-"+str(i),link_range=link_range_list[i])
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

file.close()

# 数据清洗，剔除不标准数据
with open('current_jobs.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
with open('current_jobs.txt', 'w', encoding='utf-8') as file:
    for line in lines:
        if len(line.split('|')) == 6:
            file.write(line)

end=time.time()
print("多线程爬虫耗时：{} s".format(end-start))
print("Exiting Main Thread")


