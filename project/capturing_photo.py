import cv2 as cv
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QFileDialog

class capturing_:
    def __init__(self,label,label2) :
        self.label=label
        self.label2=label2
        self.filename=""
        self.video_capture = cv.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

    def choose_photo(self):
        self.filename, _ = QFileDialog.getOpenFileName(None, "Select photo")

        if self.filename != '':
            pixmap = QPixmap(self.filename)
            self.label2.setPixmap(pixmap)
        
    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            width=self.label.width()
            height=self.label.height()
            ht, wh = frame.shape[:2] 
            if ht> height or wh > width:
                pixmap = pixmap.scaled(width, height)
                self.label.setPixmap(pixmap)
            else :
                self.label.setPixmap(pixmap)
        if not self.timer:
            self.timer = self.startTimer(30)

    def timerEvent(self, event):
        self.update_video()

    def captured_photo(self):
        ret, frame = self.video_capture.read()
        if ret :
            resized_frame = cv.resize(frame, (self.label2.width(), self.label2.height()), interpolation=cv.INTER_AREA)
            cv.imwrite(f"captured_photo.png", resized_frame)
            self.filename="captured_photo.png"
    
    
    def show_pic(self):
            pixmap = QPixmap(self.filename)
            resized_pixmap = pixmap.scaled(self.label2.width(), self.label2.height())
            self.label2.setPixmap(resized_pixmap)
    