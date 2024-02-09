import random
#import chardet
import csv

#def get_encoding(file):
#    with open(file, 'rb') as f:
#        return chardet.detect(f.read())['encoding']
#这个B函数有问题，把ANSI识别成KOIR-8
#
def get_random_file():
    #encoding=get_encoding("C:\\Users\\jiayu.sun\\Desktop\\data\\current_jobs.txt")
    #print(encoding)
    with open("C:\\Users\\jiayu.sun\\Desktop\\data\\current_jobs.txt",'r',encoding="ANSI") as file:
        lines=file.readlines()

    random_num=[]
    for i in range(1,3001):
        random_num.append(i)

    file1=open("file1.csv","w",newline='',encoding='ANSI')
    file2=open("file2.csv","w",newline='',encoding='ANSI')
    file3=open("file3.csv","w",newline='',encoding='ANSI')
    
    writer1=csv.writer(file1)
    writer1.writerow(["index","岗位名称","单位名称","行业"])
    writer2=csv.writer(file2)
    writer2.writerow(["index","岗位名称","单位名称","行业"])
    writer3=csv.writer(file3)
    writer3.writerow(["index","岗位名称","单位名称","行业"])
    for i in range(1000):
        temp=random.choice(random_num)
        myindex=random_num.index(temp)
        writer1.writerow([i,lines[temp].split('|')[0],lines[temp].split('|')[1].split('|')[0]])
        random_num.pop(myindex)

    for i in range(1000):
        temp=random.choice(random_num)
        myindex=random_num.index(temp)
        writer2.writerow([i,lines[temp].split('|')[0],lines[temp].split('|')[1].split('|')[0]])
        random_num.pop(myindex)
    
    for i in range(1000):
        temp=random.choice(random_num)
        myindex=random_num.index(temp)
        writer3.writerow([i,lines[temp].split('|')[0],lines[temp].split('|')[1].split('|')[0]])
        random_num.pop(myindex)
    
    file1.close()
    file2.close()
    file3.close()
    file.close()

get_random_file()

