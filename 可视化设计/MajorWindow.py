from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from load_data import *
from sklearn.metrics.pairwise import cosine_similarity
import math
from subassembly import PieChartWidget, BarChartWindow, ScatterChartWindow, ScrollableTextBox
import matplotlib.pyplot as plt


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

        self.post_pay = QLabel(self)
        self.post_pay.move(self.mid_width, 180)
        self.post_pay.setFixedSize(300, 50)
        self.post_pay.setFont(font)

    def my_show(self, major_name):
        majordata = majordict[major_name]
        d = sorted(list(majordata.job_rates.items()), key=lambda item: item[1], reverse=True)
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
        def query_func(post: PostData):
            job = post.job
            correlation_num = majordata.job_rates[job]
            return correlation_num * post.mark

        postlist.sort(key=query_func, reverse=True)
        temp = []
        for i, data in enumerate(postlist):
            temp.append('岗位名:' + data.name + '\n公司:' + data.recName + '\n学历要求:'
                        + data.degreeName + '\n' + f'薪资{data.lowMonthPay}-{data.highMonthPay}（千元）\n')
        # self.post_label.setText(temp)
        post_text = ScrollableTextBox(self, temp)
        post_text.move(self.mid_width, 320)
        post_text.show()
        # 显示薪资分布
        def query_func1(post: PostData):
            correlation_num = 0
            for (job, rate) in post.rates.items():
                correlation_num += rate * majordata.job_rates[job] / 100
            return correlation_num

        corrs = [query_func1(post) for post in postlist if post.MonthPay != 0 and post.mark < 28]
        corrs_sum = sum(corrs)
        pays = [post.MonthPay for post in postlist if post.MonthPay != 0]
        pay = sum([pays[i] * corrs[i] for i in range(len(corrs))]) / corrs_sum
        self.post_pay.setText('预期应届薪资:' + str(pay * 12000).split('.')[0] + '元')
        p_x, p_y = self.get_plot(corrs, 50)
        marks = [query_func(post) for post in postlist if post.MonthPay != 0 and post.mark < 28]
        # salarys = [post.MonthPay for post in postlist if post.MonthPay != 0]
        s = [min(15, post.MonthPay) for post in postlist if post.MonthPay != 0 and post.mark < 28]
        # data = [(query_func1(post), post.MonthPay) for post in postlist if post.MonthPay != 0]
        # b = ScatterChartWindow(self, data, '薪资-相关图')
        # b.move(0, 1000)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        fig, ax1 = plt.subplots(figsize=(16, 10))
        sc = ax1.scatter(corrs, marks, c=s, cmap='viridis')
        ax1.set_xlabel('相关程度')
        ax1.set_ylabel('理想程度', color='blue')
        # ax1.legend('岗位', loc='upper left')
        # plt.plot(p_x, p_y)
        ax2 = ax1.twinx()
        # ax2.set_ylabel('岗位数量', color='red')
        ax2.plot(p_x, p_y, color='red', linestyle='-', marker='o', label='Line Plot')
        colorbar = fig.colorbar(sc, ax=ax1)
        colorbar.set_label('岗位平均薪资（单位：千）')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(['岗位数量'], loc='upper left')
        # plt.subplots_adjust(right=)  # 调整右边界的位置
        plt.title('岗位信息图')
        plt.show()

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

    def get_plot(self, corr, step=10):
        s = (max(corr) - min(corr)) / step
        all_values = [0 for _ in range(step)]
        for c in corr:
            all_values[min(step - 1, int(c / s))] += 1
        all_keys = [0.5 * s + s * i for i in range(step)]
        return all_keys, all_values

    def close(self):
        plt.close()
        super().close()
