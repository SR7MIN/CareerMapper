from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from load_data import *
from sklearn.metrics.pairwise import cosine_similarity

from subassembly import PieChartWidget, BarChartWindow


class MajorWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('专业信息展示')
        self.setFixedSize(1000, 1000)
        self.mid_width = 600
        font = QFont("微软雅黑", 10, QFont.Bold)

        # 控件
        self.details_lable = QLabel(self)
        self.details_lable.move(self.mid_width, 20)
        self.details_lable.setFixedSize(350, 50)
        self.details_lable.setFont(font)

        self.salary_label = QLabel(self)
        self.salary_label.move(self.mid_width, 70)
        self.salary_label.setFixedSize(300, 50)
        self.salary_label.setFont(font)

        self.need_label = QLabel(self)
        self.need_label.move(self.mid_width, 130)
        self.need_label.setFixedSize(300, 50)
        self.need_label.setFont(font)

        self.post_label = QLabel(self)
        self.post_label.move(self.mid_width, 320)
        self.post_label.setFixedSize(400, 700)


    def my_show(self, major_name):
        majordata = majordict[major_name]
        d = sorted(list(majordata.job_rates.items()), key=lambda item:item[1], reverse=True)
        temp_name = []
        temp_rate = []
        for name, rate in d[:5]:
            temp_name.append(name)
            temp_rate.append(rate)
        a = PieChartWidget(self, '专业去向比例(前5)', temp_name, temp_rate)
        self.details_lable.setText('专业名:' + major_name)
        self.salary_label.setText('预期平均薪资:' + str(majordata.salary).split('.')[0] + '元')
        if majordict[major_name].need > 0.:
            self.need_label.setText('预期城镇行业需求：' + str(majordata.need).split('.')[0] + '人')
        else:
            self.need_label.setText('预期城镇行业需求：' + '饱和')
        # self.job_rate_label.setText(temp)
        self.set_top_related_majors(5, major_name)
        # 显示岗位
        def query_func(post:PostData):
            job = post.job
            correlation_num = majordata.job_rates[job]
            return correlation_num * post.mark

        postlist.sort(key=query_func, reverse=True)
        temp = '岗位推荐(前5)\n\n'
        for i, data in enumerate(postlist[:5]):
            temp += ('{0}.岗位名:'.format(i+1) + data.name + '\n' + '  公司:' + data.recName + '\n'
                     + '  学历要求:' + data.degreeName + '\n' + '\n')
        self.post_label.setText(temp)

        self.show()

    def set_top_related_majors(self, top_num, name):
        idx = m2i_dict[name]
        data = cosine_similarity(major_job_vectors)
        ans = data[idx]
        sorted_nums = sorted(enumerate(ans), key=lambda x: x[1], reverse=True)
        ans_idx = [i[0] for i in sorted_nums]
        data_dict = {}
        for idx in ans_idx[1:top_num + 1]:
            data_dict[i2m_dict[idx]] = float(ans[idx])
        a = BarChartWindow(self, data_dict, '就业去向相关性(前{0})'.format(top_num))
        a.move(0, 400)




