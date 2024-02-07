#用于将细分行业人数转化为国统局18个大行业人数

import pandas as pd
import csv

#需要改成本地文件地址
df=pd.read_csv("C:\\Users\\jiayu.sun\\Desktop\\data\\363个专业的信息.csv",encoding='GB2312')
df2=pd.read_csv("C:\\Users\\jiayu.sun\\Desktop\\data\\48个行业的信息.csv",encoding='GB2312')

big_industry_name_dict={0:"采矿业",1:"农林牧渔业",2:"制造业",3:"电力、燃气及水的生产和供应业",4:"建筑业",5:"交通运输、仓储及邮电通信业",6:"教育业",7:"金融业",8:"居民服务、修理和其他服务业",9:"科学研究、技术服务和地质勘查业",
                    10:"批发和零售业",11:"水利、环境和公共设施管理业",12:"卫生、社会保障和社会福利业",13:"文化、体育和娱乐业",14:"信息传输、计算机服务和软件业",15:"住宿和餐饮业",16:"租赁和商务服务业",
                    17:"房地产业"}
big_industry_name_dict2={"采矿业":0,"农林牧渔业":1,"制造业":2,"电力、燃气及水的生产和供应业":3,"建筑业":4,"交通运输、仓储及邮电通信业":5,"教育业":6,"金融业":7,"居民服务、修理和其他服务业":8,"科学研究、技术服务和地质勘查业":9,
                    "批发和零售业":10,"水利、环境和公共设施管理业":11,"卫生、社会保障和社会福利业":12,"文化、体育和娱乐业":13,"信息传输、计算机服务和软件业":14,"住宿和餐饮业":15,"租赁和商务服务业":16,
                    "房地产业":17}
small_industry_name_dict={}
major_name_dict={}

for i in range(df2.shape[0]):
    s1=df2.iloc[i,1]
    if s1!="其他行业":
        s2=df2.iloc[i,4]
        small_industry_name_dict[s1]=s2

for i in range(df.shape[0]):
    s=df.iloc[i,1]
    major_name_dict[s]=list(big_industry_name_dict.values())

for key,value in major_name_dict.items():
    for j in range(len(value)):
        value[j]=value[j]+":0"

for i in range(df.shape[0]):
     for j in range(3,df.shape[1]): 
        s=df.iloc[i,j] 
        number=s.split(':')[1]
        number=float(number) 
        name=s.split(':')[0]
        
        temp1=small_industry_name_dict[name].split('/')[0]
        flag=1
        try:
            temp2=small_industry_name_dict[name].split('/')[1]
        except IndexError:
            flag=0
        if flag==1:
            temp2=small_industry_name_dict[name].split('/')[1]
            numsum=float(major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp2]].split(':')[1])
            numsum+=number/2
            major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp2]]=major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp2]].split(':')[0]+":"+str(numsum)
            numsum=float(major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]].split(':')[1])
            numsum+=number/2
            major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]]=major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]].split(':')[0]+":"+str(numsum)
        else:
            numsum=float(major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]].split(':')[1])
            numsum+=number
            major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]]=major_name_dict[df.iloc[i][1]][big_industry_name_dict2[temp1]].split(':')[0]+":"+str(numsum)
        
print(major_name_dict)

fieldnames=major_name_dict.keys()
with open('363个专业对应大行业数据.csv','w',newline='') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(next(iter(major_name_dict.values())))):
        row = {k: v[i] for k, v in major_name_dict.items()}
        writer.writerow(row)

        

        
        







