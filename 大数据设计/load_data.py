import csv
import os

import numpy as np


def str2num(str):
    temp = str.split('-')
    if len(temp) == 2:
        return (float(temp[0].split('人')[0]) + float(temp[1].split('人')[0])) / 2
    else:
        return float(temp[0].split('人')[0])


class Data:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.people = 0
        self.rate_items = {}
        self.salary = 0

    def set_people(self, people):
        self.people = float(people)

    def set_salary(self, s):
        self.salary = float(s)


    def add_people(self, n):
        self.people = self.people + float(n)

    def set_rates(self, name, rate):
        if 'None' in name and '其他行业' in name:
            i = name.index('None')
            j = name.index('其他行业')
            rate[j] = str(float(rate[i]) + float(rate[j]))
            name.pop(i)
            rate.pop(i)
        for i in range(len(name)):
            self.rate_items[name[i]] = float(rate[i])

    def get(self):
        pass


class DataList:
    def __init__(self):
        self.dataList = []
        self.num = 0

    def find_data(self, name):
        for data in self.dataList:
            if data.name == name:
                return data
        return None

    def get_name2index_dict(self):
        outp = {}
        for data in self.dataList:
            outp[data.name] = data.id - 1
        return outp

    def get_index2name_dict(self):
        outp = {}
        for data in self.dataList:
            outp[data.id - 1] = data.name
        return outp

    def output(self):
        pass

    def load(self):
        pass


class MajorData(Data):
    def __init__(self, id, name):
        super().__init__(id, name)
        self.need_people = 0

    def set_need_people(self, str):
        self.need_people = float(str)

    def get(self):
        outlist = [self.id, self.name, self.people, self.need_people, self.salary]
        for job, rate in self.rate_items.items():
            outlist.append(job + ":" + str(rate))
        return outlist

    def get_job_names(self):
        outp = []
        for job, rate in self.rate_items.items():
            outp.append(job)
        return outp


class JobData(Data):
    def __init__(self, id, name):
        super().__init__(id, name)
        self.subjob = ''
        self.all_people = 0

    def get(self, is_sub=False):
        if is_sub:
            outlist = [self.id, self.name, self.people, self.all_people, self.salary]
        else:
            outlist = [self.id, self.name, self.people, self.subjob, self.salary]
        for major, rate in self.rate_items.items():
            outlist.append(major + ":" + str(rate))
        return outlist


    def set_subjob(self, j):
        self.subjob = j


class MajorList(DataList):
    def __init__(self):
        super().__init__()

    def get_most_salary(self, num):
        temp = self.dataList.copy()
        temp.sort(key=lambda x: x.salary, reverse=True)
        return [(data.name, data.salary) for data in temp[:num]]

    def get_least_salary(self, num):
        temp = self.dataList.copy()
        temp.sort(key=lambda x: x.salary, reverse=False)
        return [(data.name, data.salary) for data in temp[:num]]

    def set_job_rate_data(self, name, job, rate):
        data = self.find_data(name)
        if data is None:
            self.num = self.num + 1
            data = MajorData(self.num, name)
            self.dataList.append(data)
        data.set_rates(job, rate)

    def set_people(self, name, n):
        data = self.find_data(name)
        if data is None:
            self.num = self.num + 1
            data = MajorData(self.num, name)
            self.dataList.append(data)
        data.set_people(n)

    def load_job_rate_data(self, file, name):
        job = []
        rate = []
        with open(file, 'r') as f:
            line = f.readline()
            if line.replace('\r', '').replace('\n', '') == '无':
                return
            for _j in line.replace('\r', '').replace('\n', '').split(','):
                job.append(_j)
            line = f.readline()
            for _r in line.replace('\r', '').replace('\n', '').split(','):
                rate.append(_r)
        self.set_job_rate_data(name, job, rate)

    def load_from_csv(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                job = []
                rate = []
                temp = line.replace('\r', '').replace('\n', '').split(',')
                self.num = max(self.num, int(temp[0]))
                data = MajorData(int(temp[0]), temp[1])
                data.set_people(temp[2])
                data.set_need_people(temp[3])
                data.set_salary(temp[4])
                for item in temp[5:]:
                    if item != '':
                        job.append(item.split(':')[0])
                        rate.append(item.split(':')[1])
                data.set_rates(job, rate)
                self.dataList.append(data)

    def load_people_data(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.replace('\r', '').replace('\n', '').split(',')
                name = temp[0]
                if temp[1] == '':
                    continue
                people = str2num(temp[1])
                self.set_people(name, people)

    def output(self, dir=''):
        name = os.path.join(dir, '{0}个专业的信息.csv'.format(self.num))
        # path = os.path.join(self.outputdir, name)
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "专业名", "人数", "需求人数", "薪资水平", "行业情况"])
            for data in self.dataList:
                writer.writerow(data.get())

    def get_all_curr_job(self):
        all_jobs = []
        for data in self.dataList:
            jobs = data.get_job_names()
            for job in jobs:
                if job not in all_jobs:
                    all_jobs.append(job)
        return all_jobs

    def remove_incomplete(self):
        for data in self.dataList.copy():
            if len(data.rate_items) == 0 or data.people == 0:
                self.dataList.remove(data)
        self.num = 0
        for data in self.dataList:
            self.num += 1
            data.id = self.num

    def renew_job_rate(self, query_dict):
        assert len(query_dict) == len(self.get_all_curr_job()) + 1
        for data in self.dataList:
            old_j_r_dict = data.rate_items.copy()
            data.rate_items.clear()
            for job in old_j_r_dict.key():
                if job == '其他行业':
                    data.rate_items[job] = old_j_r_dict[job]
                else:
                    data.rate_items[query_dict[job]] = 0
            for job, rate in old_j_r_dict.items():
                if job != '其他行业':
                    data.rate_items[query_dict[job]] += old_j_r_dict[job]


class JobList(DataList):
    def __init__(self, is_sub=False):
        super().__init__()
        self.is_sub = is_sub

    def set_salary(self, name, salary):
        data = self.find_data(name)
        if data is None:
            return
        data.salary = salary

    def load_salary(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                temp = line.replace('\r', '').replace('\n', '').split(',')
                name = temp[0]
                salary = float(temp[1])
                self.set_salary(name, salary)

    def get_all_subjob(self):
        sub_jobs = []
        for data in self.dataList:
            if data.subjob not in sub_jobs and data.subjob != '':
                sub_jobs.append(data.subjob)
        return sub_jobs

    def set_all_people(self, name, all_people):
        data = self.find_data(name)
        if data is None:
            return
        data.all_people = all_people

    def set_people(self, name, people):
        data = self.find_data(name)
        data.set_people(people)

    def set_major_rate_data(self, name, majors, rates):
        data = self.find_data(name)
        if data is None:
            self.num = self.num + 1
            data = JobData(self.num, name)
            self.dataList.append(data)
        data.set_rates(majors, rates)

    def add_jobs(self, jobs):
        for job in jobs:
            data = self.find_data(job)
            if data is None and job != '其他行业':
                self.num = self.num + 1
                data = JobData(self.num, job)
                self.dataList.append(data)

    def output(self, dir=''):
        if self.is_sub:
            name = os.path.join(dir, '{0}个国统局行业的信息.csv'.format(self.num))
        else:
            name = os.path.join(dir, '{0}个行业的信息.csv'.format(self.num))
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            if self.is_sub:
                writer.writerow(["id", "行业名", "人数", "行业总人数（万人）", "平均薪资", "专业关联"])
            else:
                writer.writerow(["id", "行业名", "人数", "国统局对应", "平均薪资", "专业关联"])
            for data in self.dataList:
                writer.writerow(data.get(self.is_sub))

    def add_major_rate(self, job_name, major_name, major_people, rate):
        data = self.find_data(job_name)
        if data is None:
            self.num = self.num + 1
            data = JobData(self.num, job_name)
            self.dataList.append(data)
        if major_name not in data.rate_items:
            data.rate_items[major_name] = major_people * rate
        else:
            data.rate_items[major_name] += major_people * rate

    def normlise_major_rate(self, job):
        data = self.find_data(job)
        if data is None:
            pass
        majors = data.rate_items.keys()
        rates = data.rate_items.values()
        rate_sum = sum(rates)
        for m in majors:
            data.rate_items[m] /= rate_sum

    def load_from_csv(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                majors = []
                rates = []
                temp = line.replace('\r', '').replace('\n', '').split(',')
                self.num = max(self.num, int(temp[0]))
                data = JobData(int(temp[0]), temp[1])
                data.set_people(temp[2])
                data.set_salary(temp[4])
                if self.is_sub:
                    data.all_people = float(temp[3])
                else:
                    data.set_subjob(temp[3])
                for item in temp[5:]:
                    if item != '':
                        majors.append(item.split(':')[0])
                        rates.append(item.split(':')[1])
                data.set_rates(majors, rates)
                self.dataList.append(data)

    def load_all_people(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                temp = line.replace('\r', '').replace('\n', '').split(',')
                name = temp[0]
                all_people = float(temp[1])
                self.set_all_people(name, all_people)


def load_train_data(datadir, data_name, people_name, job_name):
    mlist = MajorList()
    mlist.load_from_csv(os.path.join(datadir, data_name))
    mlist.load_people_data(os.path.join(datadir, people_name))
    mlist.remove_incomplete()
    jlist = JobList()
    jlist.load_from_csv(os.path.join(datadir, job_name))
    i2n_dict_major = mlist.get_index2name_dict()
    i2n_dict_job = jlist.get_index2name_dict()
    n2i_dict_major = mlist.get_name2index_dict()
    n2i_dict_job = jlist.get_name2index_dict()
    dicts = [i2n_dict_major, i2n_dict_job, n2i_dict_major, n2i_dict_job]
    output = []
    i2list_dict = {}
    for i in range(mlist.num):
        temp = [0 for _ in range(jlist.num)]
        major_name = i2n_dict_major[i]
        data = mlist.find_data(major_name)
        all_job_index = [n2i_dict_job[job] for job in data.rate_items.keys() if job != '其他行业']
        count = 0
        for j in range(jlist.num):
            if j in all_job_index:
                temp[j] = data.rate_items[i2n_dict_job[j]]
                count += 1
        init_rate = data.rate_items['其他行业'] / (jlist.num - count)
        for j in range(jlist.num):
            if temp[j] == 0:
                temp[j] = init_rate
        output.append(temp)
        i2list_dict[i] = all_job_index
    output = np.asarray(output) / 100
    return output, i2list_dict, dicts


def output_train_data(datadir, data, dicts, peopledir):
    i2n_dict_major, i2n_dict_job, n2i_dict_major, n2i_dict_job = dicts
    mlist = MajorList()
    for i, l in enumerate(data):
        name = i2n_dict_major[i]
        jobs, rates = [], []
        for j, rate in enumerate(l):
            jobs.append(i2n_dict_job[j])
            rates.append(rate*100)
        mlist.set_job_rate_data(name, jobs, rates)
    mlist.load_people_data(peopledir)
    mlist.output(datadir)



# datadir = './major'

# # datafiles = os.listdir(datadir)
# # for file in datafiles:
# #     path = os.path.join(datadir, file)
# #     dlist.load_job_rate_data(path, file.split('.')[0])

#
#
# jlist = JobList()
# jlist.load_from_csv("D:\\大创\ForContest\\大数据设计\\data\\48个行业的信息.csv")
# jlist.add_jobs(dlist.get_all_curr_job())
# jlist.output()


# dlist = MajorList()
# dlist.load_from_csv("D:\\大创\\ForContest\\大数据设计\\logs\\000\\363个专业的信息.csv")
# dlist.load_people_data("D:\\大创\\ForContest\\大数据设计\\专业人数.csv")
# dlist.remove_incomplete()
# dlist.output('D:\\大创\\ForContest\\大数据设计\\data')
