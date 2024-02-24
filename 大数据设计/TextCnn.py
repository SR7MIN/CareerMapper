import csv

import gensim
import numpy as np
import torch
from gensim.models import Word2Vec
from torch import Tensor, nn
import os
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from model_helper import TextCnnModel
from load_text_data import load_text_data, load_apply_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
# 全局变量
basedir = './logs'  # 训练数据保存文件夹
expname = '1003'  # 实验名


def create_textcnn_model():
    # 模型参数设定
    embedding_dim = 100
    output_dim = 47
    out_channels = 66
    filter_sizes = (3, 4, 5)

    # 训练参数设定
    lrate = 5e-4  # 学习率
    start = 0  # 训练起点轮数
    raw_noise_std = 0.00  # 为了正规化输出sigma_a而添加的噪声的标准差（也可为0）

    # 初始化模型参数
    model = TextCnnModel(20, embedding_dim, output_dim, out_channels, filter_sizes).to(device)
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
    i_weights = 5000  # 保存训练信息的轮数
    i_test = 5000
    writer = SummaryWriter(os.path.join(basedir, expname))

    w2v_model = gensim.models.Word2Vec.load("./words.model")
    inputs, targets, label_list, l2i_dict, i2l_dict, sentence_list = load_text_data("111.csv", w2v_model, 20, 100,
                                                                                    False)

    rand_num = 1600
    test = Tensor(np.stack(inputs[:rand_num])).to(device)
    test_tar = np.stack(targets[:rand_num])
    inputs = Tensor(np.stack(inputs)).to(device)
    targets = Tensor(np.stack(targets)).to(device)

    model, start, grad_vars, optimizer, raw_noise_std = create_textcnn_model()
    lose_func = nn.CrossEntropyLoss()
    global_step = start

    N_iters = 10000 + 1
    batch = 1024
    start = start + 1
    i_batch = 0

    apply = False

    # 开始训练

    for i in trange(start, N_iters):
        input = inputs[i_batch:i_batch + batch][:, None]
        target = targets[i_batch:i_batch + batch]

        i_batch += batch
        if i_batch >= inputs.shape[0]:
            # 重新打乱
            # print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(inputs.shape[0])
            inputs = inputs[rand_idx]
            targets = targets[rand_idx]
            i_batch = 0

        output = model(input)

        if not apply:
            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(target.shape) * raw_noise_std
                noise = torch.Tensor(noise).to(device)

            optimizer.zero_grad()
            loss = lose_func(output, target + noise)
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
                outs = model(test[:, None]).cpu().numpy()
                with open(testsavedir, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['岗位', '预测分类', '实际分类'])
                    fault_count = 0
                    for i, out in enumerate(outs):
                        idx_p = int(np.where(out == max(out))[0])
                        idx_t = int(np.where(test_tar[i] == max(test_tar[i]))[0])
                        csv_writer.writerow([sentence_list[i], i2l_dict[idx_p], i2l_dict[idx_t]])
                        if idx_p != idx_t:
                            fault_count += 1
                            if idx_t in [6, 21] and idx_p in [6, 21]:
                                fault_count -= 1
                    csv_writer.writerow(['误差：{}%'.format(fault_count / rand_num * 100)])

        global_step += 1


def apply(path, save_path):
    w2v_model = gensim.models.Word2Vec.load("./words.model")
    text_list, inputs, i2l_dict = load_apply_data(path, w2v_model, 20, 100)
    inputs = Tensor(np.stack(inputs)).to(device)
    model, start, grad_vars, optimizer, raw_noise_std = create_textcnn_model()
    testsavedir = os.path.join(save_path, '预测结果.csv')
    with torch.no_grad():
        outs = model(inputs[:, None]).cpu().numpy()
        with open(testsavedir, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['岗位', '企业', '预测分类'])
            for i, out in enumerate(outs):
                idx_p = int(np.where(out == max(out))[0])
                csv_writer.writerow([text_list[i][0], text_list[i][1], i2l_dict[idx_p]])


# train()
# apply('all_jobs.txt', os.path.join(basedir, expname))
