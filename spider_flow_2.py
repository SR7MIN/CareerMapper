from bs4 import BeautifulSoup
import requests
import json
import jsonpath
def get_people_num_step_1(addr):
    headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    response=requests.get(addr,headers=headers)
    str=response.content.decode()
    return str

def file_write(file,jsonobj):
    file.write("".join(jsonpath.jsonpath(jsonobj,"$.msg.zymc")))
    file.write(" ")
    file.write("".join(jsonpath.jsonpath(jsonobj,"$.msg.xsgm")))
    file.write("\n")
    
def get_people_num():
#1
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73381107?_t=1706953980412')
    jsonobj=json.loads(str)
    file=open("各专业人数.txt","w")
    file_write(file,jsonobj)
#2
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384336?_t=1706966202842')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#3
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384356?_t=1706967458318')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#4
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383751?_t=1706967572563')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#5    
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383907?_t=1706967761665')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#6   
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383960?_t=1706967828627')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#7
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383203?_t=1706967873315')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#8
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384088?_t=1706968795170')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#9
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383599?_t=1706967917058')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#10
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73381155?_t=1706967969818')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#11
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383179?_t=1706968008459')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#12
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383455?_t=1706968042080')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#13
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383287?_t=1706968073845')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#14
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383191?_t=1706968104870')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#15
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384076?_t=1706968143426')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#16
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383587?_t=1706968192966')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#17
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384328?_t=1706968235505')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#18
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383847?_t=1706968274319')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#19
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383819?_t=1706968356332')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#20
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384156?_t=1706968408829')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#21
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73384204?_t=1706968449160')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#22
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383799?_t=1706968481179')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
#23
    str=get_people_num_step_1('https://gaokao.chsi.com.cn/zyk/zybk/specialityDetail/73383775?_t=1706968882314')
    jsonobj=json.loads(str)
    file_write(file,jsonobj)
    file.close()

get_people_num()
