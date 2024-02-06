import csv
import os
import numpy as np


def load_salary(file):
    i2n_dict, n2i_dict = {}, {}
    count = 0
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            i2n_dict[count] = temp[0]
            n2i_dict[temp[0]] = count
            count += 1
            dataline = [float(str) for str in temp[2:]]
            data.append(dataline)
    return np.asarray(data).swapaxes(0, 1), i2n_dict, n2i_dict


def load_both_indicator(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = line.replace('\r', '').replace('\n', '').split(',')
            dataline = [float(str) for str in temp[1:]]
            data.append(dataline)
    return np.asarray(data)