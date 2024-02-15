import os
from get_random import get_file
from TextCnn import apply
# 运行脚本获取所有岗位
os.system('python ./spider_job.py')
# 自动分类岗位
get_file()
apply('all_jobs.csv', '')
with open('预测结果.csv', 'r') as file:
    with open('current_jobs.txt', 'r') as f:
        csv_lines = file.readlines()[1:]
        txt_lines = f.readlines()
        assert len(csv_lines) == len(txt_lines)
        for i in range(len(csv_lines)):
            csv_temp = csv_lines[i].replace('\r', '').replace('\n', '').split(',')
            txt_lines[i] = txt_lines[i].replace('\r', '').replace('\n', '') + '|' + csv_temp[-1] + '\n'
    with open('current_jobs.txt', 'w') as f:
        f.writelines(txt_lines)