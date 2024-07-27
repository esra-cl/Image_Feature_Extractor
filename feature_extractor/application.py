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
    def __init__(self):
        super().__init__()
        self.app = Ui_application()
        self.app.setupUi(self)
        self.worker=None
        self.capturing = capturing_(self.app.camera,self.app.photo)
        self.applicate()
        self.histo_dict= None
        
        
    def applicate(self):
        self.app.choose_photo.clicked.connect(self.capturing.choose_photo)
        self.app.make_guess.clicked.connect(self.set_make_guess_button)
        
        self.app.capture_photo.clicked.connect(self.capturing.captured_photo) 
        self.app.capture_photo.clicked.connect(self.capturing.show_pic)

        self.app.clear.clicked.connect(self.clear_)
        self.app.hog.clicked.connect(self.hog_active)
        self.app.lbp.clicked.connect(self.lbp_active)
        self.app.make_guess.clicked.connect(self.get_checkbox_status)
        self.app.show_histogram.clicked.connect(self.histo_extraction)

    def closeEvent(self, event):
        self.clear_()
        event.accept()
        with open ("lbp.txt","w") as f:
            f.write('')
        f.close()
        with open ("hog.txt","w") as f:
            f.write('')
        f.close()

    def histo_extraction(self):
        h= histogram_extraction(self.app)
        h.histo_prepare(self.histo_dict)

    def get_checkbox_status(self):
        if self.app.lbp.checkState():
            self.LBP__()
        elif self.app.hog.checkState():
            file_name = os.path.basename(self.capturing.filename)
            self.Hog(file_name)
    
    def Hog(self, photo)-> str :
        h = Hog(photo)
        features_number, histo_dictionary = h.convert2_36vector()
        self.histo_dict=histo_dictionary
        return f"the number of features : {features_number}"

    def LBP__(self):
        file_name = os.path.basename(self.capturing.filename)
        self.worker = WorkerThread(file_name)
        self.worker.result_signal.connect(self.update_photo)
        self.worker.start()

    def update_photo(self,image_array,histo_dict):
        self._array= image_array
        self.histo_dict = histo_dict
        image_matrix = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
        
        # Get the dimensions of the image
        height, width, channels = image_matrix.shape
        q_image = QImage(image_array, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        if height > self.app.photo.height() or width > self.app.photo.width():
            q_image = QImage(image_array, 357, 199, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            pixmap= pixmap.scaled(357, 199)
            

        self.app.photo.setPixmap(pixmap)

    def PreparingThePhoto(self):
        imGray = cv.cvtColor(self.filename, cv.COLOR_BGR2GRAY)
        imGray=imGray.astype("float64")/255. #changing the type from unit8 to float32/float64 and scale it between 1-255 
        return imGray

    def set_make_guess_button(self):
        if self.app.hog.checkState() and self.capturing.filename !='':
            get_result = self.Hog(self.capturing.filename)
            
    
    def clear_(self):
        file_path ='histogram.png'
        try:
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        self.app.photo.clear()
        self.app.hog.setCheckState(False)
        self.app.lbp.setCheckState(False)
        self.app.histogram.clear()
        with open ("lbp.txt","w") as f:
            f.write('')
        f.close()
        with open ("hog.txt","w") as f:
            f.write('')
        f.close()
    def hog_active(self):
        self.app.lbp.setCheckState(False)

    def lbp_active(self):
        self.app.hog.setCheckState(False)


import numpy as np

class WorkerThread(QThread):
    result_signal = pyqtSignal(np.ndarray,dict)

    def __init__(self, input):
        super().__init__()
        self.input = input
    def run(self):
        lbp = LBP(self.input)
        image_array,histo = lbp.save_image()
        self.result_signal.emit(image_array,histo)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = application()
    
    window.show()
    sys.exit(app.exec_())


    