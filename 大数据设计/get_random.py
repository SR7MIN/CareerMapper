import random
#import chardet
import csv

#def get_encoding(file):
#    with open(file, 'rb') as f:
#        return chardet.detect(f.read())['encoding']
#这个B函数有问题，把ANSI识别成KOIR-8
#
def get_random_file(numbers):
    #encoding=get_encoding("C:\\Users\\jiayu.sun\\Desktop\\data\\current_jobs.txt")
    #print(encoding)
    with open("C:\\Users\\jiayu.sun\\Desktop\\data\\current_jobs.txt", 'r', encoding="ANSI") as file:
        lines=file.readlines()

    random_num=[]
    for i in range(1,len(lines)):
        random_num.append(i)

    file1=open("file1.csv","w",newline='', encoding='ANSI')
    writer1=csv.writer(file1)
    writer1.writerow(["index", "岗位名称", "单位名称", "行业"])
    for i in range(numbers):    #输入你想要获取的随机条数
        temp=random.choice(random_num)
        myindex=random_num.index(temp)
        if len((lines[temp]).split('|'))>1:
            writer1.writerow([i,lines[temp].split('|')[0],lines[temp].split('|')[1].split('|')[0]])
            random_num.pop(myindex)
        else:
            i-=1
    file1.close()


def get_file():
    with open("current_jobs.txt", 'r') as file:
        lines = file.readlines()
    
    newfile = open("all_jobs.csv","w",newline='')
    writer = csv.writer(newfile)
    writer.writerow(["index","岗位名称","单位名称"])
    for i in range(len(lines)):
        lineparts=lines[i].split('|')
        if len(lineparts)>1:
            writer.writerow([i,lineparts[0],lineparts[1].split('|')[0]if '|' in lineparts[1] else lineparts[1]])
    newfile.close()

# get_random_file(25000)
# get_file()

