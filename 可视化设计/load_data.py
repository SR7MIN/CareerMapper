import os
from pypinyin import lazy_pinyin
all_majors = []
all_jobs = []
all_subs = []

majordict = {}
jobdict = {}
subdict = {}

postlist = []

m2i_dict = {}
i2m_dict = {}
i2j_dict = {}
j2i_dict = {}
i2p_dict = {}
p2i_dict = {}
major_job_vectors = []

all_salary_dict = {}


class MajorData:
    def __init__(self, people, need, salary):
        self.people = float(people)
        self.need = float(need)
        self.salary = float(salary)
        self.job_rates = {}
        self.sub_rates = {}

    def set_job_rate(self, jobs):
        self.job_rates = jobs

    def set_sub_rate(self, subs):
        self.sub_rates =subs

class JobData:
    def __init__(self, people, sub):
        self.people = float(people)
        self.sub = sub
        self.major_rates = {}

    def set_major_rate(self, rates):
        self.major_rates = rates

class SupData:
    def __init__(self, major_people, people, salary):
        self.major_people = major_people
        self.people = people
        self.salary = salary
        self.major_rates = {}

    def set_major_rate(self, rates):
        self.major_rates = rates

class PostData:
    def __init__(self, name, recName, highMonthPay, lowMonthPay, headCount ,degreeName, job, mark):
        self.name = name
        self.recName = recName
        self.highMonthPay = highMonthPay
        self.lowMonthPay = lowMonthPay
        self.headCount = headCount
        self.degreeName = degreeName
        self.job = job
        self.mark = float(mark)

    def get(self):
        return '岗位名：' + self.name + ' 公司名: ' + self.recName + '学历: ' + self.degreeName




def load_major_data(dir):
    # 加载所有专业，细分和国统行业的名字，专业到行业的去向比例
    file1 = os.path.join(dir, '363个专业的信息(细分行业).csv')
    file2 = os.path.join(dir, '363个专业的信息.csv')
    file3 = os.path.join(dir, '18个国统局行业的信息.csv')
    file4 = os.path.join(dir, '47个行业的信息.csv')
    file5 = os.path.join(dir, '去向拟合数据.csv')
    file6 = os.path.join(dir, 'posts.csv')
    file7 = os.path.join(dir, '国统局薪资水平.csv')
    with open(file1, 'r') as f:
        count = 0
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            all_majors.append(temp[1])
            m2i_dict[temp[1]] = count
            i2m_dict[count] = temp[1]
            count += 1
            majordict[temp[1]] = MajorData(temp[2], temp[3], temp[4])
            rates = {}
            for s in temp[5:]:
                a, b = s.split(':')
                rates[a] = float(b)
            majordict[temp[1]].set_job_rate(rates)
    with open(file2, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            rates = {}
            for s in temp[5:]:
                a, b = s.split(':')
                rates[a] = float(b)
            majordict[temp[1]].set_sub_rate(rates)
    with open(file3, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            all_subs.append(temp[1])
            subdict[temp[1]] = SupData(temp[2], temp[3], temp[4])
            rates = {}
            for s in temp[5:]:
                a, b = s.split(':')
                rates[a] = float(b)
            subdict[temp[1]].set_major_rate(rates)
    with open(file4, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            all_jobs.append(temp[1])
            jobdict[temp[1]] = JobData(temp[2], temp[3])
            rates = {}
            for s in temp[5:]:
                a, b = s.split(':')
                rates[a] = float(b)
            jobdict[temp[1]].set_major_rate(rates)

    with open(file5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            major_job_vectors.append([float(s) for s in temp])

    with open(file6, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            if len(temp) == 8:
                postlist.append(PostData(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]))

    with open(file7, 'r') as f:
        lines = f.readlines()
        years = lines[0].replace('\r', '').replace('\n', '').split(',')[1:]
        for line in lines[1:]:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            name = temp[0]
            temp = [_ for _ in reversed(temp[1:])]
            temp_dict = {}
            for i, year in enumerate(reversed(years)):
                temp_dict[year] = float(temp[i])
            all_salary_dict[name] = temp_dict

    all_subs.sort(key=lambda char: lazy_pinyin(char)[0][0])
    all_majors.sort(key=lambda char: lazy_pinyin(char)[0][0])
    all_jobs.sort(key=lambda char: lazy_pinyin(char)[0][0])

