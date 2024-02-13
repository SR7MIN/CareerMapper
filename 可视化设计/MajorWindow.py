from PyQt5.QtWidgets import *
from load_data import *
from sklearn.metrics.pairwise import cosine_similarity

class MajorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('专业信息展示')
        self.setFixedSize(640, 320)

        # 控件
        self.details_lable = QLabel(self)
        self.details_lable.move(20, 20)
        self.details_lable.setFixedSize(350, 50)

        self.job_rate_label = QLabel(self)
        self.job_rate_label.move(20, 70)
        self.job_rate_label.setFixedSize(350, 200)

        self.salary_label = QLabel(self)
        self.salary_label.move(400, 20)
        self.salary_label.setFixedSize(250, 50)

        self.need_label = QLabel(self)
        self.need_label.move(400, 80)
        self.need_label.setFixedSize(250, 50)

        self.correlation_label = QLabel(self)
        self.correlation_label.move(400, 140)
        self.correlation_label.setFixedSize(250, 150)



    def my_show(self, major_name):
        majordata = majordict[major_name]
        d = sorted(list(majordata.job_rates.items()), key=lambda item:item[1], reverse=True)
        temp = '专业去向比例(前5)\n'
        for name, rate in d[:5]:
            temp += name + ':' + str(rate)[:4] + '%' + '\n'
        self.details_lable.setText('专业名:' + major_name)
        self.salary_label.setText('预期平均薪资:' + str(majordata.salary).split('.')[0] + '元')
        if majordict[major_name].need > 0.:
            self.need_label.setText('预期城镇行业需求：' + str(majordata.need).split('.')[0] + '人')
        else:
            self.need_label.setText('预期城镇行业需求：' + '饱和')
        self.job_rate_label.setText(temp)
        self.set_top_related_majors(5, major_name)
        self.show()

    def set_top_related_majors(self, top_num, name):
        idx = m2i_dict[name]
        data = cosine_similarity(major_job_vectors)
        ans = data[idx]
        sorted_nums = sorted(enumerate(ans), key=lambda x: x[1], reverse=True)
        ans_idx = [i[0] for i in sorted_nums]
        temp = '就业去向相关性(前{0})\n'.format(top_num)
        for idx in ans_idx[1:top_num + 1]:
            temp += i2m_dict[idx] + ':' + str(float(ans[idx]))[:5] + '\n'
        self.correlation_label.setText(temp)



