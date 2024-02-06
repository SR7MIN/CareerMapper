import csv
import numpy as np
import torch
from torch import Tensor
import os
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from model_helper import RegLSTM, MSE
from 大数据设计.load_train_data import load_people

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
basedir = './logs'  # 训练数据保存文件夹
expname = '101'  # 实验名


def create_Subjob_predictive_model(ch):
    # 模型参数设定
    netdepth = 6
    netwidth = 32

    # 训练参数设定
    lrate = 5e-4  # 学习率
    start = 0  # 训练起点轮数
    raw_noise_std = 0.002  # 为了正规化输出sigma_a而添加的噪声的标准差（也可为0）

    # 初始化模型参数
    model = RegLSTM(ch, netwidth, netdepth).to(device)
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


def train():
    # 测试变量
    i_print = 10000  # 打印测试信息的轮数
    i_weights = 500  # 保存训练信息的轮数
    i_test = 100
    writer = SummaryWriter(os.path.join(basedir, expname))
    # 人员数据到2022结束
    data, i2n_dict, n2i_dict = load_people("./data\\job\\国统局行业人数.csv")
    init_data = data.copy()
    data = data[1:] - data[:data.shape[0]-1]

    # test = np.concatenate([indicators[1], np.asarray([data.shape[0] + 1])])
    # 归一化
    scaler_max = 1
    scaler_list = [[data[:, i].min(), data[:, i].max() - data[:, i].min()] for i in range(len(i2n_dict))]
    for i in range(len(i2n_dict)):
        data[:, i] = (data[:, i] - scaler_list[i][0]) / scaler_list[i][1] * scaler_max
    # _max = data.max()
    # _min = data.min()
    # data = (data - data.min()) / (data.max() - data.min()) * scaler_max

    date_num = data.shape[0]
    targets = data.copy()[:date_num - 1]
    inputs = data.copy()[1:]

    targets = torch.Tensor(targets[:, None]).to(device)
    inputs = torch.Tensor(inputs[:, None]).to(device)
    test = targets[0:]

    model, start, grad_vars, optimizer, raw_noise_std = create_Subjob_predictive_model(data.shape[1])
    global_step = start

    N_iters = 10000 + 1
    start = start + 1

    apply = True


    # 开始训练
    for i in trange(start, N_iters):

        output = model(inputs)

        if not apply:
            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(targets.shape) * raw_noise_std
                noise = torch.Tensor(noise).to(device)

            optimizer.zero_grad()
            loss = MSE(output, targets + noise)

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
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {MSE(output[:output.shape[0] - 1], targets[1:])}")

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
                out = model(test[1:])  # 以20年为时间步预测
                loss = torch.sqrt(MSE(output[:output.shape[0] - 1], targets[1:])).cpu().numpy()
                _data = init_data[1:] - init_data[:init_data.shape[0] - 1]
                # output = output / scaler_max * (_max - _min) + _min
                out = torch.stack([out[..., i] / scaler_max * scaler_list[i][1] + scaler_list[i][0] for i in range(out.shape[-1])], dim=-1)
                output = out[-1].reshape(test.shape[-1])
                with open(testsavedir, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['行业', '预测数据', '理论误差范围（正负）'])
                    for i, num in enumerate(output):
                        csv_writer.writerow([i2n_dict[i], num.cpu().numpy() + init_data[-1][i], abs(loss * np.percentile(sorted(_data[..., i]), 50))])



        global_step += 1


train()