import random

import jieba
import numpy as np
from gensim.models import Word2Vec

sentence_list = []
label_list = []
l2i_dict = {}
i2l_dict = {}

inputs = []
targets = []


def load_train_data(data_path):
    with open("训练标签.csv", 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            l2i_dict[temp[0]] = count
            i2l_dict[count] = temp[0]
            count += 1
    labels = list(l2i_dict.keys())
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            if temp[1] in labels:
                label_list.append(temp[1])
                sentence_list.append(temp[0])


def load_all_jobs(data_path):
    with open(data_path, 'r', encoding='ANSI') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            if len(temp) >= 3:
                sentence_list.append(temp[1] + temp[2])


def sentence2word():
    words_list = []
    jieba.add_word('社招')
    jieba.add_word('秋招')
    jieba.add_word('校招')
    for sentence in sentence_list:
        words_list.append(list(jieba.cut(sentence)))
    return words_list


def word2data(words_list, data_dir):
    output_len = len(i2l_dict)
    global inputs
    inputs = np.load(data_dir)
    for i, words in enumerate(words_list):
        tar_vec = np.zeros(output_len)
        tar_vec[l2i_dict[label_list[i]]] = 1
        targets.append(tar_vec)


def load_text_data(data_path, word_dir, _rand=False):
    load_train_data(data_path)
    words_list = sentence2word()
    word2data(words_list, word_dir)
    if _rand:
        inp = inputs
        tar = targets
        x = [[inp[i], tar[i]] for i in range(len(inputs))]
        random.shuffle(x)
        inp = [x[i][0] for i in range(len(x))]
        tar = [x[i][1] for i in range(len(x))]
        return inp, tar, label_list, l2i_dict, i2l_dict, sentence_list
    return inputs, targets, label_list, l2i_dict, i2l_dict, sentence_list


def load_apply_data(data_npy, data_csv, model: Word2Vec):
    # 提供两个input和一个text_list（岗位文本），i2l_dict（编号转行业）
    with open("训练标签.csv", 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            i2l_dict[count] = temp[0]
            count += 1
    text_list = []
    # 以下读取数据1
    inputs1 = np.load(data_npy)
    # 以下构造数据2
    inputs2 = []
    with open(data_csv, 'r',encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')

            if len(temp) >= 3:
                text_list.append([temp[1], temp[2]])
                sentence_list.append(temp[1] + temp[2])

    words_list = sentence2word()
    all_words = model.wv.index_to_key
    for i, words in enumerate(words_list):
        temp = np.zeros([20, 100])
        for j in range(min(len(words), 20)):
            if words[j] in all_words:
                temp[j] = np.asarray(model.wv[words[j]])
        inputs2.append(temp)
    inputs2 = np.asarray(inputs2, dtype=np.float32)
    return text_list, inputs1, inputs2, i2l_dict



# train_word_vector("./all_jobs.txt")

# load_train_data("data/岗位训练数据集.csv")
# sentence2word()
# model = word2dict()
# v = model.wv['汽车']
