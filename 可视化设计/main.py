from PyQt5.Qt import *
import sys
from MainWindow import MainWindow

if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())