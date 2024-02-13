from PyQt5.QtWidgets import *
from load_data import *

class JobWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('行业信息展示')
        self.setFixedSize(640, 320)

        # 控件
        self.details_lable = QLabel(self)
        self.details_lable.move(20, 20)
        self.details_lable.setFixedSize(350, 50)

        self.major_rate_label = QLabel(self)
        self.major_rate_label.move(20, 70)
        self.major_rate_label.setFixedSize(350, 200)

        self.salary_label = QLabel(self)
        self.salary_label.move(400, 20)
        self.salary_label.setFixedSize(250, 50)

        self.people_label = QLabel(self)
        self.people_label.move(400, 150)
        self.people_label.setFixedSize(250, 50)


    def my_show(self, name, cater):
        self.details_lable.setText('行业名:' + name)
        if cater == 'sub':
            jobdata = subdict[name]
        else:
            jobdata = jobdict[name]
        d = sorted(list(jobdata.major_rates.items()), key=lambda item: item[1], reverse=True)
        temp = '行业相关度(前5)\n'
        for name, rate in d[:5]:
            temp += name + ':' + str(rate)[:4] + '%' + '\n'
        self.major_rate_label.setText(temp)
        if cater == 'sub':
            self.salary_label.setText('预期平均薪资:' + str(jobdata.salary).split('.')[0] + '元')
        else:
            sub_name = jobdata.sub
            self.salary_label.setText('预期平均薪资:' + str(subdict[sub_name].salary).split('.')[0] + '元\n'
                                      + '(来源于' + sub_name + ')')
        self.people_label.setText('行业人数：' + str(jobdata.people).split('.')[0] + '人')
        self.show()

