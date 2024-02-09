import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch import Tensor
from tqdm import trange, tqdm

from load_data import load_train_data, output_train_data, MajorList, JobList

from torch.utils.tensorboard import SummaryWriter

from model_helper import Major_Rate_Model, MSE

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
basedir = './logs'  # 训练数据保存文件夹
expname = '000'  # 实验名


def create_major_relateed_model(input_ch, output_ch):
    '''
    不可训练，废弃
    '''
    # 模型参数设定
    netdepth = 4
    netwidth = 128

    # 训练参数设定
    lrate = 5e-4  # 学习率
    start = 0  # 训练起点轮数
    raw_noise_std = 1e0  # 为了正规化输出sigma_a而添加的噪声的标准差（也可为0）

    # 初始化模型参数
    model = Major_Rate_Model(netdepth, netwidth, input_ch, output_ch).to(device)
    grad_vars = list(model.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    # 加载检查点
    check_points = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname)))
                    if 'tar' in f]
    if len(check_points) > 0:
        print('Found ckpts', check_points)
        check_point_path = check_points[-1]
        print('Reloading from', check_point_path)
        check_point = torch.load(check_point_path)
        # 加载训练数据
        start = check_point['global_step']
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        # 加载模型
        model.load_state_dict(check_point['network_fn_state_dict'])

    return model, start, grad_vars, optimizer, raw_noise_std


def my_func(output, value, data, _list):
    _out_sum = 0
    _tar_sum = 0
    target = Tensor(len(output)).to(device)
    for i in range(len(output)):
        if i not in _list:
            _out_sum += data[i]
            _tar_sum += value[i]
            target[i] = value[i]
    target *= (_out_sum / _tar_sum)
    for i in _list:
        target[i] = data[i]
    return MSE(output, target)


# noise = 0.
# if raw_noise_std > 0.:
#     noise = torch.randn(raw[..., 1].shape) * raw_noise_std


def train_major_related_model():
    '''
        训练专业相关模型，废弃
    '''
    # 测试变量
    i_print = 100  # 打印测试信息的轮数
    i_testset = 5000  # 导出测试集的轮数
    i_weights = 1000  # 保存训练信息的轮数

    writer = SummaryWriter(os.path.join(basedir, expname))

    data, i2list_dict = load_train_data('./data', '363个专业的信息.csv', '专业人数.csv', '47个行业的信息.csv')
    data = torch.Tensor(data).to(device)
    major_num = data.shape[0]
    job_num = data.shape[1]
    model, start, grad_vars, optimizer, raw_noise_std = create_major_relateed_model(major_num, job_num)

    N_iters = 30001 + 1
    print('Begin')
    global_step = start
    start = start + 1
    input = torch.eye(major_num).to(device)
    related_vectors = []
    for i in range(major_num):
        _data = torch.concat([data[:i], data[i + 1:]], 0)
        related_vector = torch.cosine_similarity(_data, data[i][None, :].expand_as(_data))
        related_vectors.append(torch.softmax(related_vector, 0))

    for i in trange(start, N_iters):
        major_index = i % major_num
        time0 = time.time()

        _data = torch.concat([data[:major_index], data[major_index + 1:]], 0)
        output = model(input[major_index])
        value = torch.matmul(related_vectors[major_index], _data)

        optimizer.zero_grad()
        loss = my_func(output, value, data[major_index], i2list_dict[major_index])
        loss.backward()  # 损失反向传播
        optimizer.step()  # 调用优化器的 step() 方法更新模型参数

        # 动态更新学习率
        lrate_decay = 250
        lrate = 5e-4
        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        writer.add_scalar("模型的loss变化", loss.item(), i)
        dt = time.time() - time0

        # 打印训练信息
        if i % i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")

        # 保存训练参数
        if i % i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 导出测试
        if i % i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.csv'.format(i))
            with torch.no_grad():
                test = torch.eye(major_num).to(device)
                outputs = []
                for l in test:
                    outputs.append(model(l))
                with open(testsavedir, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['生成数据'])
                    csv_writer.writerows(outputs)
                    csv_writer.writerow(['原始数据'])
                    csv_writer.writerows(data)

            print('Saved test set')

        global_step += 1


def fit_major_rate_by_related(is_output=False):
    '''
    基于专业相关性拟合就业去向数据
    '''
    data, i2list_dict, dicts = load_train_data('./data', '363个专业的信息.csv', '专业人数.csv', '47个行业的信息.csv')
    init_data = data.copy()
    data = torch.Tensor(data).to(device)
    major_num = data.shape[0]
    job_num = data.shape[1]

    related_vectors = []
    for i in range(major_num):
        _data = torch.concat([data[:i], data[i + 1:]], 0)
        related_vector = torch.cosine_similarity(_data, data[i][None, :].expand_as(_data))
        related_vectors.append(torch.softmax(related_vector, 0))

    N_iter = 25 + 1

    for i in trange(1, N_iter):

        mark = 0
        for major_index in range(major_num):
            # 计算专业相关性向量
            _data = torch.concat([data[:major_index], data[major_index + 1:]], 0)
            related_vector = torch.cosine_similarity(_data, data[i][None, :].expand_as(_data))
            related_vector = torch.softmax(related_vector, 0)
            # 计算拟合就业向量
            value = torch.matmul(related_vector, _data)
            # 计算分数
            _list = i2list_dict[major_index]
            _len = job_num - len(i2list_dict[major_index])
            a, b = [], []
            for _i in range(job_num):
                if _i not in _list:
                    a.append(value[_i])
                    b.append(data[major_index][_i])
            a, b = Tensor(a).reshape([1, -1]), Tensor(b).reshape([1, -1])
            mark += torch.cosine_similarity(a, b)
            # 更新填充数据
            a *= b.sum() / a.sum()
            flag = 0
            for _i in range(job_num):
                if _i not in _list:
                    data[major_index][_i] = a[0][flag]
                    flag += 1

        # 打印训练信息
        tqdm.write(f"[TRAIN] Iter: {i} Mark: {float(mark)}")

    # 导出数值
    if is_output:
        testsavedir = os.path.join(basedir, expname, '拟合数据.csv')
        with open(testsavedir, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['生成数据'])
            csv_writer.writerows(data.cpu().numpy())
            csv_writer.writerow(['原始数据'])
            csv_writer.writerows(init_data)

    output_train_data(os.path.join(basedir, expname), data.cpu().numpy(), dicts, './data/专业人数.csv')

    return data.cpu().numpy()


def convert_mrate2jrate(mlist: MajorList, jlist: JobList):
    job_list = mlist.get_all_curr_job()
    assert jlist.num == len(job_list)
    for job in job_list:
        majors, rates, people = [], [], 0
        for data in mlist.dataList:
            if job in data.rate_items:
                majors.append(data.name)
                rates.append(data.people * data.rate_items[job])
                people += data.people * data.rate_items[job] / 100
        rates = [r / sum(rates) * 100 for r in rates]
        jlist.set_major_rate_data(job, majors, rates)
        jlist.set_people(job, people)
    return jlist


def convert_j2subj(jlist: JobList):
    all_subjob = jlist.get_all_subjob()
    subjlist = JobList(True)
    majors = [_ for _ in jlist.dataList[0].rate_items.keys()]
    for subjob in all_subjob:
        rates, peoples, people_num = [], [], 0
        for data in jlist.dataList:
            if data.subjob == subjob:
                rates.append([_ for _ in data.rate_items.values()])
                peoples.append(data.people)
                people_num += data.people
        rates, peoples = np.asarray(rates), np.asarray(peoples)
        peoples /= sum(peoples)
        rates = np.multiply(rates, peoples[:, None])
        rates = np.sum(rates, 0)
        subjlist.set_major_rate_data(subjob, majors, rates)
        subjlist.set_people(subjob, people_num)
    return subjlist


def convert_salary_job2major(jlist: JobList, mlist: MajorList):
    for m_data in mlist.dataList:
        salary = 0
        for j_data in jlist.dataList:
            salary += j_data.salary / 100 * m_data.rate_items[j_data.name]
        m_data.set_salary(salary)
    mlist.output()


def get_job_need(file, dir):
    name, need = [], []
    with open(file, 'r') as f:
        _ = f.readline()
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            name.append(temp[0])
            need.append((float(temp[1]) - float(temp[2])) * 10000)
    filepath = os.path.join(dir, '行业需求情况.csv')
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(name)):
            writer.writerow([name[i], need[i]])


def get_major_need(file, mlist: MajorList, jlist: JobList):
    all_job = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            all_job[temp[0]] = float(temp[1])
    for m_data in mlist.dataList:
        need = 0
        for j_data in jlist.dataList:
            need += j_data.rate_items[m_data.name] / 100 * all_job[j_data.name]
        m_data.set_need_people(need)
    mlist.output()


mlist = MajorList()
mlist.load_from_csv("363个专业的信息.csv")
# l = mlist.get_least_salary(10)

jlist = JobList()
jlist.load_from_csv("D:\\大创\ForContest\\大数据设计\\data\\47个行业的信息.csv")
# s = jlist.get_all_subjob()
j = convert_j2subj(jlist)
j.load_all_people("D:\\大创\\ForContest\\大数据设计\\data\\job\\国统局行业人数.csv")
j.load_salary("D:\\大创\\ForContest\\大数据设计\\data\\job\\国统局薪资水平.csv")
j.output('D:\\大创\\ForContest\\大数据设计\\data')
# jlist = convert_mrate2jrate(mlist, jlist)
# jlist.output('D:\\大创\\ForContest\\大数据设计')
convert_salary_job2major(j, mlist)
mlist.output()
# jlist = JobList(True)
# jlist.load_from_csv("data\\18个国统局行业的信息.csv")
# get_job_need("data\\job\\国统局行业人数.csv", '')
# get_major_need("行业需求情况.csv", mlist, jlist)

