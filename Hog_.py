import cv2 as cv 
from PIL import Image
import numpy as np 
from PIL import Image
import numpy as np
import math 


class Hog:
    def __init__(self,file_name):
        self.file_name= file_name
        self.magnitude_cells=[]
        self.direction_cells=[]
        self.columns=None
        self.rows=None
        self.histograms = []

    def resize_image(self):
        image = Image.open(self.file_name)
        resized_image =image.resize((64,128))
        resized_image= np.array (resized_image,dtype=np.float32)           
        self.rows=len(resized_image[:,0]) - 7
        self.columns= len(resized_image[0,:]) - 7
        resized_image =cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        print("the max value of the resized image",np.max(resized_image))
        return resized_image
        
    def scaling(self):
        scaled = np.zeros((128, 64), dtype=np.float32)
        image_matrix = self.resize_image().astype(np.float32) 
        print("shape of the image after resizing ",image_matrix.shape)
        for i in range(self.rows+7):
            for j in range(self.columns+7):
                scaled[i,j] = image_matrix[i,j]/255.0 
        print("shape of the scalled matrix", scaled.shape)
        return scaled

    def conv_(self,rows,columns,padded_iamge,kernal):
        _row= []
        new_val= 0.0
        res=[]
        for i in range(rows):
            for j in range(columns):
                for k in range(kernal.shape[0]):
                    for l in range(kernal.shape[1]):
                        new_val +=padded_iamge[i+k,j+l]*kernal[k,l]
                _row.append(new_val)
                new_val=0.0
            res.append(_row)
            _row=[]
        
        res= np.array(res)
        #print("shape of the resulted matrix ", res.shape)
        return res
    
    def clc_gradinatxy(self):
        scaled = self.scaling()
        #x gradiantı hesapla
        x_kernal =np.array([[1,0,-1]])
        #padding----------------------------------------------------
        column = np.zeros((128,1),dtype=np.float32)  
        padded = np.concatenate((column, scaled, column), axis=1)
        print("the shape of the columns pixels padded imag",padded.shape)
        #----------------------------------------------------
        rows1 = len(padded[:,0]) 
        columns1 = len(padded[0,:])- (x_kernal.shape[1])+1
        gx = self.conv_(rows1,columns1,padded,x_kernal)

        #y gradiantı hesapla----------------------------------------
        y_kernal = np.array([[1],[0],[-1]])
        #padding ---------------------------------------------------
        row = np.zeros((1, 64), dtype=np.float32) 
        padded_ = np.concatenate((row, scaled, row), axis=0)  
        print("the shape of rows padded image ",padded_.shape)
        #----------------------------------------------------
        rows2 = len(padded_[:,0]) - (y_kernal.shape[0])+1
        columns2 = len(padded_[0,:])
        gy = self.conv_(rows2,columns2,padded_,y_kernal)
        return gx,gy
    
    def magnitude_clc(self,gx,gy):
        magnitude= math.sqrt(gx**2 + gy**2)
        return magnitude
    
    def direction_clc(self,gx,gy):
        theta = math.atan2(gy, gx)
        _degrees = math.degrees(theta)
        _degrees=_degrees % 180
        return _degrees
    
    def magnitude_direction_clc(self):
        gx,gy = self.clc_gradinatxy()                
        print("gradiantx after convulating the vertical kernal",gx.shape)
        print(f"gradianty after conulating the horizontal kernal ",gy.shape)
        magnitude_matrix = np.zeros((gx.shape[0], gx.shape[1]), dtype=np.float32)
        direction_matrix = np.zeros((gx.shape[0], gx.shape[1]), dtype=np.float32)

        for i in range(gx.shape[0]):  # Iterate over rows
            for j in range(gx.shape[1]):  # Iterate over columns
                magnitude_matrix[i, j] = self.magnitude_clc(gx[i, j], gy[i, j])
                direction_matrix[i, j] = self.direction_clc(gx[i, j], gy[i, j])
        return magnitude_matrix,direction_matrix
    
    def portion_clc(self,rate,mag):
        portion1 = rate*mag
        portion2 = (1-rate)*mag
        return portion1,portion2
    
    def extracte_vector(self,histo):
        vector ={}
        sum =0
        for key , item in histo.items() :
            for it in item:
                sum+=it
            vector[key]=sum
            sum=0
        for key, item in vector.items():
            #print(key,item)
            pass
        return vector
    
    def histograms_extractions(self,directions,magnintude ):
        histograms = []
        print(directions.shape)
        for i in range(directions.shape[0]):
            #print(directions[i].shape)
            direc_list = directions[i].tolist()
            mag_list= magnintude[i].tolist()
            #print("the size of the magnitude cell",len(mag_list),len(magnintude[0]))
            histogram = self.histogram_clc(mag_list,direc_list)    
            histo = self.extracte_vector(histogram)
            for key,i in histo.items():
                histograms.append(i)  
        histograms=np.array([histograms])
        histograms = histograms.reshape(8,16,9)
        return histograms

    def histogram_clc(self,mag,direc):
        #her columnda en fazla 64 tane deger gelebilir  keylerimiz 0dan 160a kadar acılar olacak 
        histogram = {angle: [] for angle in range(0, 161, 20)}

        vector =[0,20,40,60,80,100,120,140,160]

        for i in range(len(mag)):
            rate = 0.0
            por1,por2=0.0,0.0
            for j in range(len(mag[0])):
                for k in range(len(vector)):
                    if direc[i][j]<vector[k] and vector[k]>0:
                        left=vector[k]-20
                        right= vector[k]
                        diff1 = direc[i][j] - left 
                        diff2= right - direc[i][j] 
                        #print("left, right , vectorek",left,right,vector[k],"directions,dif1,dif2",direc[i][j],diff1,diff2)
                        if diff2>diff1 : 
                            rate = diff1/(diff1 + diff2)
                            por1,por2= self.portion_clc(rate,mag[i][j])
                            histogram[left].append(por2)
                            histogram[right].append(por1)
                        else:                         
                            rate = diff2/(diff1 + diff2) 
                            por1,por2= self.portion_clc(rate,mag[i][j])   
                            histogram[right].append(por2)
                            histogram[left].append(por1)    
                        break
                    elif direc[i][j]==vector[k]:
                        histogram[vector[k]].append(mag[i][j])  
                    elif direc[i][j]>160:
                        left =160
                        #print("left",left)
                        right= 180      # 180 
                        diff1 = direc[i][j] - left #179-160  = 19
                        diff2= right - direc[i][j] #180-179 = 1  
                        right=0                    # left = 160 right = 0
                        #print("-------------------------------->left, right , vectorek",left,right,"directions,dif1,dif2",direc[i][j],diff1,diff2)
                        if diff2>diff1 : 
                            rate = diff1/(diff1 + diff2)
                            por1,por2= self.portion_clc(rate,mag[i][j])
                            histogram[left].append(por2)
                            histogram[right].append(por1)
                        else:                            #19>1
                            rate = diff2/(diff1 + diff2) # rate = 1/20 
                            por1,por2= self.portion_clc(rate,mag[i][j])    #por1,por2= 1/(20)*7 , (1-1/(20)*7)  
                            histogram[right].append(por2) # histogram[0] = por2 = 6.65 , histogram[160 ]= por1 = 0.35
                            histogram[left].append(por1)  
                        break  

#        for ind ,ele in  histogram.items():
#            print(ind,ele)
#            print("-------------------")
        return histogram

    def divid_2cells (self):
        magnitude_matrix,direction_matrix=self.magnitude_direction_clc()
        magnitude_row_=[]
        magnitude_cell = []
        direction_row_=[]
        direction_cell = []
        #print("rows",self.rows,"columns",self.columns )
        for i in range(0,self.rows,8):
            for j in range(0,self.columns,8):
                for k in range(8):
                    for l in range(8):
                        magnitude_row_.append(magnitude_matrix[i+k][j+l])
                        direction_row_.append(direction_matrix[i+k][j+l])
                    magnitude_cell.append(magnitude_row_)
                    direction_cell.append(direction_row_)
                    magnitude_row_=[]
                    direction_row_=[]
                self.magnitude_cells.append(magnitude_cell)
                self.direction_cells.append(direction_cell)
                direction_cell=[]
                magnitude_cell=[]
        self.magnitude_cells= np.array(self.magnitude_cells)
        self.direction_cells=np.array(self.direction_cells)
        
        histograms = self.histograms_extractions(self.direction_cells, self.magnitude_cells)
        
        print("max of direction cells ",np.max(self.direction_cells))
        print("shape of the magnitude cells ",self.magnitude_cells.shape)
        print("shape of the direction cells ",self.direction_cells.shape)
        
        print("direction first cell ",self.direction_cells[0,:,:])
        print("magnitude first cell ",self.magnitude_cells[0,:,:])
        print("\n\n")
        return histograms
      
    def normalization (self,vector):
        instensity = 0
        for bin in vector :
            instensity +=bin**2
        instensity=math.sqrt(instensity)

        for idx in range(len(vector)):
            vector[idx] = vector[idx]/instensity
        return vector
    
    def convert2_36vector(self):
        histogram_for4cells = []
        histogram_=[]
        histogram= self.divid_2cells()
        print("shape of histogram", histogram.shape)
        for i in range (histogram.shape[0]-1):
            for j in range ( histogram.shape[1]-1):
                for k in range(2):
                    for l in range(2):
                        for bin in histogram[(i+k),(j+l),:]:
                            histogram_for4cells.append(bin)
                vector = self.normalization(histogram_for4cells)
                histogram_.append(vector)
                histogram_for4cells=[]                
        histogram_= np.array(histogram_)
        histogram_=histogram_.reshape(7,15,36)
        print("the shape of the histo", histogram_.shape)
        histogram_=histogram_.reshape(3780)
        final_histogram= self.histogram_extracte(histogram_)
        return histogram_,final_histogram
    
    def histogram_extracte(self,histo_array):
        histo_dict={}
        histo = histo_array.tolist()
        f = open("hog.txt", "w")


        for i in range(len(histo)):
            histo_dict[i]= histo[i]
            f.write(f"key:{i}, value:{histo[i]}\n")
        f.close()
        return histo_dict


        
        
                        


            


        

