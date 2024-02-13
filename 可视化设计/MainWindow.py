from load_data import *
from PyQt5.QtWidgets import *
from MajorWindow import MajorWindow
from JobWindow import JobWindow


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("就业信息查询系统")
        self.setFixedSize(640, 320)
        load_major_data('./data')

        # 控件
        self.Major_label = QComboBox(self)
        self.Major_label.setFixedSize(300, 50)
        self.Major_label.move(100, 20)
        self.Major_label.addItems(all_majors)

        self.Job_label = QComboBox(self)
        self.Job_label.setFixedSize(300, 50)
        self.Job_label.move(100, 120)
        self.Job_label.addItems(all_jobs + all_subs)

        self.Major_button = QPushButton(self)
        self.Major_button.setText('查询专业')
        self.Major_button.setFixedSize(100, self.Major_label.size().height())
        self.Major_button.move(self.Major_label.x() + self.Major_label.size().width() + 10, self.Major_label.y())

        self.Job_button = QPushButton(self)
        self.Job_button.setText('查询行业')
        self.Job_button.setFixedSize(100, self.Job_label.size().height())
        self.Job_button.move(self.Job_label.x() + self.Job_label.size().width() + 10, self.Job_label.y())

        # 窗口
        self.Major_window = MajorWindow()
        self.Job_window = JobWindow()

        # 信号
        self.Major_button.clicked.connect(self.show_major_window)
        self.Job_button.clicked.connect(self.show_job_window)

    def show_major_window(self):
        self.Major_window.my_show(self.Major_label.currentText())

    def show_job_window(self):
        name = self.Job_label.currentText()
        if name in all_subs:
            self.Job_window.my_show(name, 'sub')
        else:
            self.Job_window.my_show(name, 'job')


