import numpy as np
import csv
import pandas as pd
from assess_enterprise import assess_enterprise_point

average_point=assess_enterprise_point(2024)

def assess_job():
#按照Saaty层次分析法标度标准计算权重：
#            min max degree enterprise
#       min   1  1/3   2       1/2
#       max   3   1    6        2
#    degree  1/2 1/6   1       1/2
#enterprise   2  1/2   2        1
#该矩阵经一致性检验后CR约为0.02<0.1,其不一致程度在可接受范围内
    MATRIX=np.array([[1,1/3,2,1/2],
                   [3,1,6,2],
                   [1/2,1/6,1,1/2],
                   [2,1/2,2,1]])
    SUM=pow(MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]*MATRIX[0][3],1/3)+pow(MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]*MATRIX[1][3],1/3)+pow(MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]*MATRIX[2][3],1/3)+pow(MATRIX[3][0]*MATRIX[3][1]*MATRIX[3][2]*MATRIX[3][3],1/3)
    MIN_SALARY_POINT=pow((MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]*MATRIX[0][3]),1/3)/SUM
    MAX_SALARY_POINT=pow((MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]*MATRIX[1][3]),1/3)/SUM
    POINT_DEGREE=pow((MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]*MATRIX[2][3]),1/3)/SUM
    ENTERPRISE_POINT=pow((MATRIX[3][0]*MATRIX[3][1]*MATRIX[3][2]*MATRIX[3][3]),1/3)/SUM

    DEGREE_DICT={"不限\n":"0","专科及以上\n":"1","本科及以上\n":"2","硕士及以上\n":"3","博士及以上\n":"4"}
    
    #debug
    print(MIN_SALARY_POINT)
    print(MAX_SALARY_POINT)
    print(POINT_DEGREE)
    print(ENTERPRISE_POINT)
    #如何解决所有的无分数企业问题
    path="C:\\Users\\jiayu.sun\\Desktop\\CareerMapper-main\\enterprise_rate.csv"
    PD=pd.read_csv(path,encoding='ANSI')
    ENTERPRISE_DICT=dict(zip(PD["企业"],PD["评分"]))

    #debug
    #print(ENTERPRISE_DICT["学信咨询服务有限公司"])

    POINT=[]

    with open("current_jobs.txt", 'r', encoding="utf-8") as file:
        lines=file.readlines()
    
    file=open("jod_rate.csv","w",newline='',encoding='ANSI')
    writer=csv.writer(file)
    writer.writerow(["岗位名称","评分"])
    #计算分数
    for i in range(len(lines)):
        fields=lines[i].split('|')
        if len(lines[i].split('|'))==6&len(fields[5])!=0:
            name=fields[0]
            enterprise=fields[1]
            max_salary_str=fields[2]
            min_salary_str=fields[3]
            degree_str=fields[5]
            try:
                point=float(min_salary_str)*MIN_SALARY_POINT+float(max_salary_str)*MAX_SALARY_POINT+float(DEGREE_DICT[degree_str])*POINT_DEGREE+float(ENTERPRISE_DICT[enterprise])*ENTERPRISE_POINT
            except KeyError:
                point=float(min_salary_str)*MIN_SALARY_POINT+float(max_salary_str)*MAX_SALARY_POINT+float(DEGREE_DICT[degree_str])*POINT_DEGREE+average_point*ENTERPRISE_POINT
            finally:
                POINT.append(point)
                writer.writerow([name,point])    
    file.close()

assess_job()

