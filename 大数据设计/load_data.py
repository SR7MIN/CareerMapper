import csv
import os


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

    def set_people(self, people):
        self.people = float(people)

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

    def output(self):
        pass

    def load(self):
        pass


class MajorData(Data):
    def __init__(self, id, name):
        super().__init__(id, name)

    def get(self):
        outlist = [self.id, self.name, self.people]
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
        self.salary = 0

    def get(self):
        outlist = [self.id, self.name, self.people, self.salary]
        for major, rate in self.rate_items.items():
            outlist.append(major + ":" + str(rate))
        return outlist

    def set_salary(self, s):
        self.salary = float(s)


class MajorList(DataList):
    def __init__(self):
        super().__init__()

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
                self.num += 1
                data = MajorData(self.num, temp[1])
                data.set_people(temp[2])
                for item in temp[3:]:
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

    def output(self):
        name = '{0}个专业的信息.csv'.format(self.num)
        # path = os.path.join(self.outputdir, name)
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "专业名", "人数", "行业情况"])
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


class JobList(DataList):
    def __init__(self):
        super().__init__()

    def add_jobs(self, jobs):
        for job in jobs:
            data = self.find_data(job)
            if data is None:
                self.num = self.num + 1
                data = JobData(self.num, job)
                self.dataList.append(data)

    def output(self):
        name = '{0}个行业的信息.csv'.format(self.num)
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "行业名", "人数", "专业关联"])
            for data in self.dataList:
                writer.writerow(data.get())

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



datadir = './major'
dlist = MajorList()
# dlist.load_from_csv("C:\\Users\\86183\\Desktop\\23个专业的信息.csv")
# dlist.load_people_data("D:\\大创\\ForContest\\大数据设计\\data\\各专业人数.txt")
datafiles = os.listdir(datadir)
for file in datafiles:
    path = os.path.join(datadir, file)
    dlist.load_job_rate_data(path, file.split('.')[0])
dlist.load_people_data("D:\\大创\\ForContest\\大数据设计\\专业人数.csv")
dlist.remove_incomplete()
#
#
jlist = JobList()
jlist.add_jobs(dlist.get_all_curr_job())
jlist.output()
# dlist.output()
