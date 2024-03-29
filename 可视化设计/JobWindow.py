from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from load_data import *
from subassembly import PieChartWidget, ColoredLineChart


class JobWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('行业信息展示')
        self.setFixedSize(1500, 850)
        mid_width = 600

        font = QFont("微软雅黑", 10, QFont.Bold)

        # 控件
        self.details_lable = QLabel(self)
        self.details_lable.move(mid_width, 20)
        self.details_lable.setFixedSize(400, 100)
        self.details_lable.setFont(font)

        self.salary_label = QLabel(self)
        self.salary_label.move(mid_width, 120)
        self.salary_label.setFixedSize(600, 100)
        self.salary_label.setFont(font)

        self.people_label = QLabel(self)
        self.people_label.move(mid_width, 250)
        self.people_label.setFixedSize(600, 100)
        self.people_label.setFont(font)


    def my_show(self, name, cater):
        self.details_lable.setText('行业名:' + name)
        if cater == 'sup':
            jobdata = subdict[name]
        else:
            jobdata = jobdict[name]
        d = sorted(list(jobdata.major_rates.items()), key=lambda item: item[1], reverse=True)

        # 行业相关度
        temp_name = []
        temp_rate = []
        for _name, _rate in d[:5]:
            temp_name.append(_name)
            temp_rate.append(_rate)
        a = PieChartWidget(self, '专业相关度(前5)', temp_name, temp_rate)

        # 薪资水平
        if cater == 'sup':
            sub_name = name
            self.salary_label.setText('预期平均薪资:' + str(jobdata.salary).split('.')[0] + '元')
        else:
            sub_name = jobdata.sub
            self.salary_label.setText('预期平均薪资:' + str(subdict[sub_name].salary).split('.')[0] + '元\n'
                                      + '(来源于' + sub_name + ')')
        l = len(all_salary_dict[sub_name])
        b = ColoredLineChart(self, all_salary_dict[sub_name], [l-2, l-1], '历史薪资数据',
                             '预测薪资数据', '历年平均薪资折线图（单位：元）')
        b.move(0, 350)
        if cater == 'sup':
            self.people_label.setText('行业人数：' + str(jobdata.people).split('.')[0] + '万人')
        else:
            self.people_label.setText('行业人数：' + str(jobdata.people).split('.')[0] + '人')

        self.show()

