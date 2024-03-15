import random

from PyQt5.QtChart import QPieSeries, QPieSlice, QChart, QChartView, QCategoryAxis, QValueAxis, QScatterSeries, \
    QLineSeries, QBarSeries, QBarSet
from PyQt5.QtCore import Qt, QPointF, QPoint
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QCursor, QFont, QRadialGradient
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QGraphicsLinearLayout, \
    QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QTextEdit, QScrollBar


def get_lower_num(num):
    num = int(num)
    i = 10000
    return num // i * i


def get_upper_num(num):
    num = int(num)
    i = 10000
    return (num // i + 2) * i


class PieChartWidget(QWidget):
    def __init__(self, parent, name, names, rates):
        super().__init__(parent)
        self.setFixedSize(600, 400)

        self.my_valueLabel = QLabel()
        self.my_valueLabel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.my_valueLabel.setFont(QFont("Comic Sans MS", 10, QFont.Bold))
        self.my_valueLabel.setWindowFlags(Qt.FramelessWindowHint)
        self.my_valueLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.my_valueLabel.hide()

        _rate = 100 - sum(rates)
        self.series = QPieSeries()
        slice0 = QPieSlice(names[0] + ':{:.2f}%'.format(rates[0]), rates[0])
        slice0.setBrush(QColor("#fa543b"))
        # slice0.setLabelVisible(True)

        slice1 = QPieSlice(names[1] + ':{:.2f}%'.format(rates[1]), rates[1])
        slice1.setBrush(QColor("#ff5733"))
        # slice1.setLabelVisible(True)
        slice2 = QPieSlice(names[2] + ':{:.2f}%'.format(rates[2]), rates[2])
        slice2.setBrush(QColor("#33ff57"))
        # slice2.setLabelVisible(True)
        slice3 = QPieSlice(names[3] + ':{:.2f}%'.format(rates[3]), rates[3])
        slice3.setBrush(QColor("#5733ff"))
        # slice3.setLabelVisible(True)
        slice4 = QPieSlice(names[4] + ':{:.2f}%'.format(rates[4]), rates[4])
        slice4.setBrush(QColor("#33a8ff"))
        # slice4.setLabelVisible(True)
        slice5 = QPieSlice("其他:{:.2f}%".format(_rate), _rate)
        slice5.setBrush(QColor("#FFB6C1"))
        # slice5.setLabelVisible(True)

        self.series.append(slice0)
        self.series.append(slice1)
        self.series.append(slice2)
        self.series.append(slice3)
        self.series.append(slice4)
        self.series.append(slice5)

        chart = QChart()
        chart.addSeries(self.series)
        chart.createDefaultAxes()
        chart.legend().setAlignment(Qt.AlignLeft)
        chart.setTitle(name)
        chart.setBackgroundBrush(QColor("white"))
        chart.legend().setVisible(True)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)

        # 悬停信号连接
        self.series.hovered.connect(self.slotSliceHoverd, Qt.QueuedConnection)

        self.layout = QVBoxLayout()
        self.layout.addWidget(chart_view)
        self.setLayout(self.layout)

    def slotSliceHoverd(self, slice: QPieSlice, state):
        if state:
            self.my_valueLabel.setText(slice.label())
            length = 10
            self.my_valueLabel.move(QCursor.pos().x() + length, QCursor.pos().y() + length)
            self.my_valueLabel.show()
            slice.setExploded(True)
        else:
            self.my_valueLabel.hide()
            slice.setExploded(False)


class ColoredLineChart(QWidget):
    def __init__(self, parent, data: dict, special_index, Pname, SPname, name):
        super().__init__(parent)
        self.setFixedSize(1500, 500)
        self.highlighted_flag = False

        self.my_valueLabel = QLabel()
        self.my_valueLabel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.my_valueLabel.setFont(QFont("Comic Sans MS", 10, QFont.Bold))
        self.my_valueLabel.setWindowFlags(Qt.FramelessWindowHint)
        self.my_valueLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.my_valueLabel.hide()

        # 创建折线图系列
        line_series = QLineSeries()
        pen = QPen()
        pen.setStyle(Qt.SolidLine)
        pen.setColor(QColor(21, 100, 255))
        pen.setWidth(4)
        label_list = []
        for x, (label, value) in enumerate(data.items()):
            line_series.append(QPointF(x, value))
            label_list.append((label, x))
        line_series.setPen(pen)

        # 创建图表并设置系列
        chart = QChart()
        chart.legend().setVisible(True)
        chart.addSeries(line_series)

        # 设置坐标轴和标题
        chart.setTitle(name)
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setAnimationOptions(QChart.NoAnimation)

        # 创建横轴和纵轴
        # 自定义x轴标签
        axis_x = QCategoryAxis()
        for (l, i) in label_list:
            axis_x.append(l, i)
        axis_x.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
        chart.addAxis(axis_x, Qt.AlignBottom)
        line_series.attachAxis(axis_x)

        lower = get_lower_num(min(data.values()))
        upper = get_upper_num(max(data.values()))
        axis_y = QValueAxis()
        axis_y.setRange(lower, upper)
        axis_y.setLabelFormat("%.0f")
        chart.addAxis(axis_y, Qt.AlignLeft)
        line_series.attachAxis(axis_y)

        # 标记预测点和非预测点
        point_series0 = QScatterSeries()  # 高光组件
        point_series1 = QScatterSeries()
        point_series2 = QScatterSeries()
        point_series_s = QScatterSeries()  # 特殊点
        point_float = QScatterSeries()
        # (高光组件)
        point_series0.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 圆形的点
        point_series0.setBorderColor(QColor(255, 182, 193, 100))  # 点边框色
        point_series0.setBrush(QBrush(QColor(255, 182, 193)))  # 点背景色
        point_series0.setMarkerSize(18)  # 点大小
        point_series0.setVisible(False)
        # (边框散点)
        point_series1.setName(Pname)
        point_series1.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 圆形的点
        point_series1.setBorderColor(Qt.black)  # 点边框色
        point_series1.setBrush(QBrush(Qt.black))  # 点背景色
        point_series1.setMarkerSize(12)  # 点大小
        # (中心散点)
        point_series2.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 圆形的点
        point_series2.setBorderColor(Qt.white)  # 点边框色
        point_series2.setBrush(QBrush(Qt.white))  # 点背景色
        point_series2.setMarkerSize(6)  # 点大小
        # (特殊点)
        point_series_s.setName(SPname)
        point_series_s.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 圆形的点
        point_series_s.setBorderColor(Qt.red)  # 点边框色
        point_series_s.setBrush(QBrush(Qt.red))  # 点背景色
        point_series_s.setMarkerSize(12)  # 点大小
        # 悬停标记点
        point_float.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 圆形的点
        point_float.setBorderColor(Qt.white)  # 点边框色
        point_float.setBrush(QBrush(Qt.white))  # 点背景色
        point_float.setOpacity(0.001)  # 透明
        point_float.setMarkerSize(12)  # 点大小
        for x, (label, value) in enumerate(data.items()):
            point_series1.append(QPointF(x, value))
            point_series2.append(QPointF(x, value))
            point_float.append(QPointF(x, value))
            if x in special_index:
                point_series_s.append(x, value)

        chart.addSeries(point_series0)
        chart.addSeries(point_series1)
        chart.addSeries(point_series_s)
        chart.addSeries(point_series2)
        chart.addSeries(point_float)

        point_series_s.attachAxis(axis_x)
        point_series0.attachAxis(axis_x)
        point_series0.attachAxis(axis_y)
        point_series_s.attachAxis(axis_y)
        point_series1.attachAxis(axis_y)
        point_series2.attachAxis(axis_y)
        point_float.attachAxis(axis_y)

        # 设置图例
        legend_markers = chart.legend().markers()
        for marker in legend_markers:
            if marker.label() == '':
                marker.setVisible(False)

        # 创建图表视图
        self.chart_view = QChartView(chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

        # 悬停信号连接
        point_float.hovered.connect(self.slotPointHoverd, Qt.QueuedConnection)

        # 布局设置
        layout = QVBoxLayout(self)
        layout.addWidget(self.chart_view)
        self.setLayout(layout)

    def slotPointHoverd(self, point: QPointF, state):
        if state:
            self.my_valueLabel.setText(str(point.y()))
            length = 15
            self.my_valueLabel.move(QCursor.pos().x() + length, QCursor.pos().y() + length)
            self.my_valueLabel.show()
            if not self.highlighted_flag:
                series = self.chart_view.chart().series()[1]
                series.clear()
                series.append(point)
                series.setVisible(True)
                self.highlighted_flag = True
                for marker in self.chart_view.chart().legend().markers():
                    if marker.label() == '':
                        marker.setVisible(False)

        else:
            self.my_valueLabel.hide()
            series = self.chart_view.chart().series()[1]
            series.setVisible(False)
            self.highlighted_flag = False


class BarChartWindow(QWidget):
    def __init__(self, parent, data, name):
        super().__init__(parent)
        self.setFixedSize(600, 600)
        self.my_valueLabel = QLabel()
        self.my_valueLabel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.my_valueLabel.setFont(QFont("Comic Sans MS", 10, QFont.Bold))
        self.my_valueLabel.setWindowFlags(Qt.FramelessWindowHint)
        self.my_valueLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.my_valueLabel.hide()

        # 创建柱状图系列
        bar_series = QBarSeries()
        # 创建柱状图数据集
        bar_set = QBarSet('')
        for (l, n) in data.items():
            bar_set << n

        # 将数据集添加到系列
        bar_series.append(bar_set)

        # 创建图表并设置系列
        chart = QChart()
        chart.addSeries(bar_series)

        # 设置坐标轴和标题
        chart.setTitle(name)
        chart.legend().setVisible(False)

        # 创建横轴和纵轴
        axis_x = QCategoryAxis()
        axis_x.setLabelsAngle(45)
        for i, l in enumerate(data.keys()):
            axis_x.append(l, i)
        axis_x.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
        chart.addAxis(axis_x, Qt.AlignBottom)
        axis_y = QValueAxis()
        axis_y.setRange(min(0.9, min(data.values())-0.1), 1)
        axis_y.setLabelFormat("%.2f")
        chart.addAxis(axis_y, Qt.AlignLeft)

        bar_series.attachAxis(axis_x)
        bar_series.attachAxis(axis_y)
        # 创建图表视图
        self.chart_view = QChartView(chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

        # 悬停信号连接
        bar_series.hovered.connect(self.slotBarHoverd, Qt.QueuedConnection)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(self.chart_view)

        self.setLayout(layout)


    def slotBarHoverd(self, state, index):
        if state:
            text = self.chart_view.chart().axes()[0].categoriesLabels()[index]
            num = self.chart_view.chart().series()[0].barSets()[0].at(index)
            self.my_valueLabel.setText(text + ':' + str(num)[:5])
            length = 15
            self.my_valueLabel.move(QCursor.pos().x() + length, QCursor.pos().y() + length)
            self.my_valueLabel.show()
        else:
            self.my_valueLabel.hide()


class ScatterChartWindow(QGraphicsView):
    def __init__(self, parent, data, name):
        super(ScatterChartWindow, self).__init__(parent)
        self.setWindowTitle(name)
        self.setFixedSize(600, 600)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.data = data
        self.plot_data()

    def plot_data(self):
        x_range = (min(self.data, key=lambda x: x[0])[0], max(self.data, key=lambda x: x[0])[0])
        y_range = (min(self.data, key=lambda x: x[1])[1], max(self.data, key=lambda x: x[1])[1])

        for point in self.data:
            x, y = point
            scaled_x = self.map_value(x, x_range, (0, self.width()))
            scaled_y = self.map_value(y, y_range, (self.height(), 0))

            ellipse = QGraphicsEllipseItem(scaled_x, scaled_y, 2, 2)  # Fix here
            ellipse.setBrush(Qt.blue)
            self.scene.addItem(ellipse)

    def map_value(self, value, from_range, to_range):
        return (value - from_range[0]) / (from_range[1] - from_range[0]) * (to_range[1] - to_range[0]) + to_range[0]


class ScrollableTextBox(QWidget):
    def __init__(self, parent, data):
        super().__init__(parent)
        self.curr_max = -1
        self.data = data
        self.initUI()


    def initUI(self):
        self.setWindowTitle('Scrollable Text Box')
        self.setFixedSize(400, 700)

        self.layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)

        self.scrollbar = QScrollBar()
        # self.scrollbar.setMaximum(100)  # 设置滚动条的最大值
        self.scrollbar.valueChanged.connect(self.scroll_text)  # 连接滚动条的滑动事件

        self.text_edit.setVerticalScrollBar(self.scrollbar)

        self.setLayout(self.layout)

        self.batch_size = 100  # 每批次加载的文本数量
        self.total_lines = 100000  # 总文本行数
        self.loaded_lines = 0  # 已加载的文本行数

        self.populate_text(0)

    def populate_text(self, pos):
        # 模拟从数据库或文件中加载文本的过程
        text_list = self.data[self.loaded_lines:self.batch_size + self.loaded_lines]
        random.shuffle(text_list)
        for i in range(self.batch_size):
            self.text_edit.append(f'第{self.loaded_lines + i + 1}个岗位\n' + text_list[i])

        self.loaded_lines += self.batch_size
        self.scrollbar.setValue(pos)

    def scroll_text(self, value):
        # 当滚动到底部时加载更多文本
        if self.curr_max < 0:
            self.curr_max = self.scrollbar.maximum()
        if value == self.curr_max:
            if self.loaded_lines < self.total_lines:
                self.populate_text(self.curr_max)
                # self.scrollbar.setMaximum(min(self.loaded_lines, self.total_lines))
                self.curr_max = self.scrollbar.maximum()
