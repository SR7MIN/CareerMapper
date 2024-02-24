import csv
import numpy as np
import pandas as pd

SCALE_DICT={"XS(微型)":"1","S(小型)":"2","M(中型)":"3","L(大型)":"4","--":"2.5"}

def assess_enterprise_point(year):
    #确定权重所用矩阵
    #          企业规模  注册资本  成立日期
    #企业规模     1        2         4       
    #注册资本     1/2      1         2
    #成立日期     1/4      1/2       1
    MATRIX=np.array([[1,2,4],
                    [1/2,1,2],
                    [1/4,1/2,1]])
    SUM=pow((MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]),1/3)+pow((MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]),1/3)+pow((MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]),1/3) 
    SCALE_POINT=pow((MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]),1/3)/SUM
    CAPITAL_POINT=pow((MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]),1/3)/SUM
    DATE_POINT=pow((MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]),1/3)/SUM
    
    #print(SCALE_POINT)
    #print(CAPITAL_POINT)
    #print(DATE_POINT)

    POINT=[]

    path="./data/企业信息数据.xlsx"
    PD=pd.read_excel(path)
    file=open("./data/enterprise_rate.csv","w",newline='',encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(["企业","评分"])
    #计算分数
    times=1
    sum_point=0
    for i in range(len(PD)):
        name=PD.iloc[i,2]
        if(PD.iloc[i,4]!="--"):
            #只考虑年份
            try:
                point=float(SCALE_DICT[PD.iloc[i,3]])*SCALE_POINT+float(PD.iloc[i,4])*CAPITAL_POINT+(year-float(str(PD.iloc[i,5]).split('-')[0]))*DATE_POINT
                POINT.append(point)
                sum_point+=point
                writer.writerow([name,point])
            except ValueError:
                point=0
                POINT.append(point)
                writer.writerow([name,point])
                print(times,"无日期")
                times+=1
        else:
            #最低注册资本
            try:
                point=float(SCALE_DICT[PD.iloc[i,3]])*SCALE_POINT+3*CAPITAL_POINT+(year-float(str(PD.iloc[i,5]).split('-')[0]))*DATE_POINT
                POINT.append(point)
                sum_point+=point
                writer.writerow([name,point])
            except ValueError:
                point=0
                POINT.append(point)
                writer.writerow([name,point])
                print(times,"无日期")
                times+=1
    average_point=sum_point/(len(PD)-times)
    file.close()
    return average_point

#average_point=assess_enterprise_point(2024)
#print(average_point)