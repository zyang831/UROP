# libraries
import PyQt5
import cv2
import pandas as pd
import numpy as np

#
import sys
from PyQt5.QtWidgets import QApplication
from MainWindow import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
