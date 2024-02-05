import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch import Tensor
from tqdm import trange, tqdm

from load_data import load_train_data, output_train_data

from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
basedir = './logs'  # 训练数据保存文件夹
expname = '000'  # 实验名


class Major_Rate_Model(nn.Module):
    def __init__(self, D, W, input_ch, output_ch):
        super(Major_Rate_Model, self).__init__()
        self.D = D  # D=8 netdepth网络的深度 ,也就是网络的层数 layers in network
        self.W = W  # W=256 netwidth网络宽度 , 也就每一层的神经元的个数 channels per layer
        self.input_ch = input_ch  # 输入维度
        self.output_ch = output_ch  # 输出维度

        Linears_list = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            Linears_list.append(nn.Linear(W, W))
        self.pts_linears = nn.ModuleList(Linears_list)

        # 在第9层添加24维度的方向数据和10维的距离数据，并且输出128维的信息
        self.out_linear = nn.Linear(W, output_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        assert len(h.shape) == 1
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        output = self.out_linear(h)
        output = self.sigmoid(output)
        return torch.softmax(output, 0)


def create_model(input_ch, output_ch):
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


def MSE(x, y):
    return torch.mean((x - y) ** 2)


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


def train():
    # 测试变量
    i_print = 100  # 打印测试信息的轮数
    i_testset = 5000  # 导出测试集的轮数
    i_weights = 1000  # 保存训练信息的轮数

    writer = SummaryWriter(os.path.join(basedir, expname))

    data, i2list_dict = load_train_data('./data',
                                                                                       '363个专业的信息.csv',
                                                                                       '专业人数.csv',
                                                                                       '47个行业的信息.csv')
    data = torch.Tensor(data).to(device)
    major_num = data.shape[0]
    job_num = data.shape[1]
    model, start, grad_vars, optimizer, raw_noise_std = create_model(major_num, job_num)

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



def method_1(is_output = False):
    data, i2list_dict, dicts = load_train_data('./data','363个专业的信息.csv','专业人数.csv','47个行业的信息.csv')
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
            a *= b.sum()/a.sum()
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

    output_train_data(os.path.join(basedir, expname), data.cpu().numpy(), dicts)

    return data.cpu().numpy()



method_1()