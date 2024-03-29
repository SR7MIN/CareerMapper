import csv

import numpy

from model_helper import Subjob_predictive_model, MSE
import torch
import numpy as np
from torch import Tensor
import os
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from load_train_data import load_salary, load_both_indicator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
basedir = './logs'  # 训练数据保存文件夹
expname = '012'  # 实验名


def run_network(gdb, gdb_sub, time, network_fn):
    # embedded_gdb = embed_gdb(gdb)
    # embedded_gdb_sub = embed_gdb_sub(gdb_sub)
    # embedded_time = embed_time(time)
    adds = [time, gdb_sub]
    return network_fn(gdb, adds)


def create_Subjob_predictive_model(output_ch):
    # L_GDB = 4
    # L_GDB_sub = 4
    # L_time = 4
    # embed_gdb, input_ch = get_embedder(L_GDB, 3)
    # embed_gdb_sub, input_0 = get_embedder(L_GDB_sub, 1)
    # embed_time, input_1 = get_embedder(L_time, 1)
    add_chs = [1, 3]

    # 模型参数设定
    netdepth = 4
    netwidth = 16

    # 训练参数设定
    lrate = 5e-4  # 学习率
    start = 0  # 训练起点轮数
    raw_noise_std = 0.001  # 为了正规化输出sigma_a而添加的噪声的标准差（也可为0）

    # 初始化模型参数
    model = Subjob_predictive_model(netdepth, netwidth, 3, add_chs, output_ch).to(device)
    grad_vars = list(model.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    def network_query_fn(gdb, gdb_sub, time, network_fn):
        return run_network(gdb, gdb_sub, time, network_fn)

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

    return model, network_query_fn, start, grad_vars, optimizer, raw_noise_std


def train():
    # 测试变量
    i_print = 10000  # 打印测试信息的轮数
    i_weights = 10000  # 保存训练信息的轮数
    i_test = 1000
    writer = SummaryWriter(os.path.join(basedir, expname))
    # 薪资数据从2022开始
    data, i2n_dict, n2i_dict = load_salary("./data\\job\\行业薪资原始数据.csv")
    # 公用指标从2023开始
    indicators = load_both_indicator("./data\\indicators\\所有行业.csv")

    targets = data.copy()
    for i in range(indicators.shape[1]):
        indicators[:, i] = (indicators[:, i] - indicators[:, i].min()) / (
                indicators[:, i].max() - indicators[:, i].min())
    _indi = indicators[1:1 + data.shape[0]]
    # num = data.shape[0]
    times = np.asarray([data.shape[0] - i - 1 for i in range(data.shape[0])])
    inputs = np.concatenate([_indi, times[:, None]], axis=1)
    test = np.concatenate([indicators[0], np.asarray([data.shape[0] + 1])])
    # 归一化
    scaler_max = 1
    scaler_list = [[targets[:, i].min(), targets[:, i].max() - targets[:, i].min()] for i in range(len(i2n_dict))]
    for i in range(len(i2n_dict)):
        targets[:, i] = (targets[:, i] - scaler_list[i][0]) / scaler_list[i][1] * scaler_max
    # _max = targets.max()
    # _min = targets.min()
    # targets = (targets - targets.min()) / (targets.max() - targets.min()) * scaler_max

    targets = torch.Tensor(targets).to(device)
    inputs = torch.Tensor(inputs).to(device)
    test = torch.Tensor(test).to(device)

    model, network_query_fn, start, grad_vars, optimizer, raw_noise_std = create_Subjob_predictive_model(data.shape[1])
    global_step = start

    N_iters = 10000 + 1
    start = start + 1

    apply = True

    # 开始训练
    for i in trange(start, N_iters):
        index = i % data.shape[0]
        gdb, gdb_sub, time = inputs[index][:3], inputs[index][3:6], inputs[index][6:]
        target = targets[index]

        if not apply:
            output = network_query_fn(gdb, gdb_sub, time, model)

            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(target.shape) * raw_noise_std
                noise = torch.Tensor(noise).to(device)

            optimizer.zero_grad()
            loss = MSE(output, target + noise)

            loss.backward()
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

        if i % i_test == 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.csv'.format(i))
            with torch.no_grad():
                # gdb, gdb_sub, time = test[:3], test[3:6], test[6:]
                # output = network_query_fn(gdb, gdb_sub, time, model)
                # #  output = output / scaler_max * (_max - _min) + _min
                # output = [output[i] / scaler_max * scaler_list[i][1] + scaler_list[i][0] for i in range(len(output))]
                # with open(testsavedir, 'w', newline='') as file:
                #     csv_writer = csv.writer(file)
                #     for i, num in enumerate(output):
                #         csv_writer.writerow([i2n_dict[i], num.cpu().numpy()])
                tests = torch.concatenate([test[None,:], inputs], 0)
                with open(testsavedir, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    outputs = []
                    for test in tests:
                        gdb, gdb_sub, time = test[:3], test[3:6], test[6:]
                        output = network_query_fn(gdb, gdb_sub, time, model).cpu().numpy()
                        output = [output[i] / scaler_max * scaler_list[i][1] + scaler_list[i][0] for i in
                                  range(len(output))]
                        outputs.append(output)
                    outputs = numpy.stack(outputs).swapaxes(0, 1)
                    for i, output in enumerate(outputs):
                        csv_writer.writerow([i2n_dict[i]] + list(output))


        global_step += 1


train()
