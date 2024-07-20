import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QLabel
from PyQt5.QtGui import QImage, QPixmap
from gui import Ui_application
from  capturing_photo import capturing_
from LBP_ import LBP
import cv2 as cv 
import os 
from Hog_ import Hog
from PyQt5.QtCore import QObject, QThread,pyqtSignal
from histogram_extracte import histogram_extraction
#pyuic5 gui.ui -o gui.py
class application(QMainWindow):