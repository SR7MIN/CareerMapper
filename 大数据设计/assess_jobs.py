import numpy as np
import csv

def assess_job():
#按照Saaty层次分析法标度标准计算权重：
#min max degree
    MATRIX=np.array([[1,1/3,3],
                   [3,1,1/2],
                   [1/3,2,1]])
    SUM=pow((MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]),1/3)+pow((MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]),1/3)+pow((MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]),1/3) 
    MIN_SALARY_POINT=pow((MATRIX[0][0]*MATRIX[0][1]*MATRIX[0][2]),1/3)/SUM
    MAX_SALARY_POINT=pow((MATRIX[1][0]*MATRIX[1][1]*MATRIX[1][2]),1/3)/SUM
    POINT_DEGREE=pow((MATRIX[2][0]*MATRIX[2][1]*MATRIX[2][2]),1/3)/SUM

    DEGREE_DICT={"不限\n":"0","专科及以上\n":"1","本科及以上\n":"2","硕士及以上\n":"3","博士及以上\n":"4"}

    #print(MIN_SALARY_POINT)
    #print(MAX_SALARY_POINT)
    #print(POINT_DEGREE)

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
            max_salary_str=fields[2]
            min_salary_str=fields[3]
            degree_str=fields[5]
            point=float(min_salary_str)*MIN_SALARY_POINT+float(max_salary_str)*MAX_SALARY_POINT+float(DEGREE_DICT[degree_str])*POINT_DEGREE
            POINT.append(point)
            writer.writerow([name,point])

    file.close()

#assess_job()

