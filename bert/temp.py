import csv
# 筛选111.csv
label_list = []
l2i_dict = {}
i2l_dict = {}
sentence_list = []

with open("训练标签.csv", 'r') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        temp = line.replace('\r', '').replace('\n', '').split(',')
        l2i_dict[temp[0]] = count
        i2l_dict[count] = temp[0]
        count += 1
labels = list(l2i_dict.keys())

with open('111.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        temp = line.replace('\r', '').replace('\n', '').split(',')
        if temp[1] in labels:
            sentence_list.append(temp)

with open('222.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    for temp in sentence_list:
        csv_writer.writerow(temp)

