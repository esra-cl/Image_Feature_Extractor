import numpy as np
import cv2

class LBP:
    def __init__(self,imag_path) :
        self.image_path=imag_path
        self.lbp_image= None

    def lbp(self, crop):
        lbp_sum=0
        R = 1
        P = 8  
        gc=crop[1][1]
        gc_x = 1
        gc_y = 1
        height, width = len(crop), len(crop)  
        for i in range(P):
            theta = 2*np.pi*i/P
            x =round(gc_x -R*np.sin(theta)) 
            y =round(gc_y +R*np.cos(theta))
            if 0 <= x<width and 0 <= y<height:
                binary_vl = self.s(crop[x][y],gc)
                lbp_sum += binary_vl*(2**i)
        return lbp_sum

    def s(self,gp,gc)->int:
        if gp >=gc:
            return 1 
        else :
            return 0 
        
    def travel_(self,photo):
        image = cv2.imread(self.image_path)  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        matrix=gray
        core_x=0
        core_y=0
        for i in range(len(matrix)-2):
            for j in range(len(matrix[0])-2):
                crop =[]
                for k in range(3):
                    crop_row =[]
                    for m in range(3):
                        crop_row.append(matrix[i+k][j+m])
                        if k==1 and m==1:
                            core_x=i+k
                            core_y=j+m
                    crop.append(crop_row)
                self.lbp(crop)
                matrix[core_x][core_y]=self.lbp(crop)
        trimmed_matrix =matrix[1:-1]  # Remove the first and last rows
        trimmed_matrix =[row[1:-1] for row in trimmed_matrix]  # Remove the first and last columns for each row
        return trimmed_matrix
    
    def histo_extracte (self,array_):
        histo= {}
        for i in range(0,256): 
            histo[i]=0
        for i in range( array_.shape[0]):
            for j in range(array_.shape[1]):
                if array_[i][j] in histo.keys():
                    histo[array_[i][j]] +=1
        f = open(fr"C:\Users\HP\Downloads\feature_extractor\Image_Feature_Extractor\Image_Feature_Extractor\lbp.txt", "w")
        for key, item in histo.items():
            histo[key] = item /(array_.shape[0]*array_.shape[1])
            f.write(f"key:{key}, value:{item /(array_.shape[0]*array_.shape[1])}\n")
        f.close()
        #----------------------------------
        


        return histo
    


    def save_image(self):
        matrix=self.travel_(self.image_path)
        image_array = np.array(matrix)
        histo = self.histo_extracte(image_array)
        print(image_array.shape)
        cv2.imwrite('result_image.png',image_array)
        return image_array,histo 
    


        




    


    
       
