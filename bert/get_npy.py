import threading

import jieba
import numpy as np
from bert_serving.client import BertClient  # 导入客户端
from tqdm import trange

sentence_list = []
inputs = []
words_list = []
curr = 0
total = 0


class myThread(threading.Thread):
    def __init__(self, name, s, e):
        threading.Thread.__init__(self)
        self.name = name
        self.s = s
        self.e = e

    def run(self):
        print("Starting" + self.name)
        word2data(20, self.s, self.e)
        print("Exiting" + self.name)


def load_all_jobs(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').replace(' ', '').split(',')
            if len(temp) >= 3:
                sentence_list.append(temp[1] + temp[2])


def load_train_data(data_path):
    label_list = []
    l2i_dict = {}
    i2l_dict = {}
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


def load_apply_data(data_path):
    # 格式 index 岗位 企业
    with open(data_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            if len(temp) >= 3:
                sentence_list.append(temp[1] + temp[2])


def sentence2word(sentence_len):
    jieba.add_word('社招')
    jieba.add_word('秋招')
    jieba.add_word('校招')
    for sentence in sentence_list:
        words_list.append(list(jieba.cut(sentence.replace(' ', '').replace('\t', '').replace('\u3000', ''))))
    global inputs
    inputs = np.zeros([len(words_list), sentence_len, 768], dtype=np.float32)
    global total
    total = len(words_list)


def word2data(sentence_len, start, end):
    bc = BertClient()
    for i in range(start, end):
        if i >= len(words_list):
            pass
        words = words_list[i]
        temp = np.zeros([sentence_len, 768])
        _len = min(len(words), sentence_len)
        temp[:_len] = np.asarray(bc.encode(words))[:_len]
        # for j in range(min(len(words), sentence_len)):
        #     if words[j] != '':
        mutex.acquire()
        inputs[i] = temp
        global curr
        curr = curr + 1
        print('\r进度：{0}/{1}'.format(curr, total), end="")
        mutex.release()



mutex = threading.Lock()
# load_all_jobs('all_jobs.csv')
# load_train_data('111.csv')
load_apply_data('all_jobs.csv')
sentence2word(20)
threads = []
all_thread = 4
num = int(total / all_thread)
link_range_list = [(i*num, (i + 1)*num) for i in range(all_thread)]
for i in range(all_thread):
    thread=myThread("Thread-"+str(i),s=link_range_list[i][0], e=link_range_list[i][1])
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()
# word2data(20, 0, 2000)
np.save('word_vectors_apply', inputs)
