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
    with open("./data/训练标签.csv", 'r') as f:
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


def word2dict(words_list):
    model = Word2Vec(
        words_list,  # 需要训练的文本
        vector_size=100,
        min_count=1,  # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
        sg=1,  # 训练方法 1：skip-gram 0；CBOW。
        epochs=100,  # 语料库上的迭代次数
        hs=1
    )
    model.save('words.model')
    return model


def word2data(model: Word2Vec, sentence_len, vector_size, words_list):
    output_len = len(i2l_dict)
    for i, words in enumerate(words_list):
        temp = np.zeros([sentence_len, vector_size])
        for j in range(min(len(words), sentence_len)):
            temp[j] = np.asarray(model.wv[words[j]])
        inputs.append(temp)
        tar_vec = np.zeros(output_len)
        tar_vec[l2i_dict[label_list[i]]] = 1
        targets.append(tar_vec)


def load_text_data(data_path, model: Word2Vec, sentence_len, vector_size):
    load_train_data(data_path)
    words_list = sentence2word()
    word2data(model, sentence_len, vector_size, words_list)
    return inputs, targets, label_list, l2i_dict, i2l_dict, sentence_list


def load_apply_data(data_path, model: Word2Vec, sentence_len, vector_size):
    with open("./data/训练标签.csv", 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            i2l_dict[count] = temp[0]
            count += 1
    text_list = []
    with open(data_path, 'r', encoding='ANSI') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            if len(temp) >= 3:
                text_list.append([temp[1], temp[2]])
                sentence_list.append(temp[1] + temp[2])
    words_list = sentence2word()
    all_words = model.wv.index_to_key
    for i, words in enumerate(words_list):
        temp = np.zeros([sentence_len, vector_size])
        for j in range(min(len(words), sentence_len)):
            if words[j] in all_words:
                temp[j] = np.asarray(model.wv[words[j]])
        inputs.append(temp)
    return text_list, inputs, i2l_dict



def train_word_vector(path):
    load_all_jobs(path)
    words_list = sentence2word()
    _ = word2dict(words_list)


# train_word_vector("./all_jobs.txt")

# load_train_data("data/岗位训练数据集.csv")
# sentence2word()
# model = word2dict()
# v = model.wv['汽车']
