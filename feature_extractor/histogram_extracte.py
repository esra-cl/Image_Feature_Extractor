import cv2 as cv 
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage,QPixmap
import numpy as np


class histogram_extraction :

    def __init__(self,app):
        self.histogram_= None
        self.app= app 


    def save_histogram(self, histo_dict, filename):
        keys = list(histo_dict.keys())
        values = list(histo_dict.values())
        first_key = next(iter(histo_dict.keys()))
        last_key = next(reversed(histo_dict.keys()))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})

        # Plotting the histogram
        bars1 = ax1.bar(keys, values, width=20, edgecolor='black')
        
        ax1.set_xticks(ticks=range(first_key,last_key, 50))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        step =(last_key - first_key)//10
        
        bins = [bin for bin in keys if bin%step==0]
        values_ = [f'{histo_dict[key]:.4f}'  for key in keys if key in bins]  # Using 

        # Adding the table
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)

        table_data = [values_, bins]
        the_table = ax2.table(cellText=table_data,
                            cellLoc='center',
                            loc='center',
                            cellColours=[['w'] * len(values_), ['w'] * len(bins)])

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.5)

        # Adjusting cell properties
        for (i, j), cell in the_table.get_celld().items():
            cell.set_width(0.1)
            cell.set_height(0.15)
            cell.set_edgecolor('black')

        plt.tight_layout()
        plt.savefig('histogram.png')
        plt.close()

    def histo_prepare(self,histo_dict):
        imag_path = "histogram.png"
        histogram_dic = self.save_histogram(histo_dict,imag_path)
        
        histogram_image = cv.imread(imag_path)
        histogram_image = cv.cvtColor(histogram_image, cv.COLOR_BGR2RGB)

        # Convert histogram image to QImage
        q_image = QImage(histogram_image.data, histogram_image.shape[1], histogram_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        width=self.app.histogram.width()
        hieght=self.app.histogram.height()
        pixmap_resized =pixmap.scaled(width, hieght)      
        # Set pixmap to label
        self.app.histogram.setPixmap(pixmap_resized)