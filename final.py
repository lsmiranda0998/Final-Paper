import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets,uic
from PyQt5.QtWidgets import QMainWindow,QFileDialog 
from PyQt5.QtGui import QPixmap
from tensorflow.keras.preprocessing import image
from telaDois import Form
from TelaInformacoes import Formu
import imutils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
modelBin = Sequential()
modelBin = load_model('try1024RedeBIN.h5')
modelBin.summary()
modelTipos = Sequential()
modelTipos = load_model('try1024RedeTIPOS.h5')
modelTipos.summary()
qtCreatorFile = "TelaPrincipal.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self)  :
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.contador= "NULL"
        self.contadorNormal = 0
        self.contadorL1 = 0
        self.contadorL2 = 0
        self.contadorL3 = 0
        self.PNormal = 0
        self.P1 = 0
        self.P2 = 0
        self.P3 = 0
        self.PTotal=0
        self.imgAux  = "a"
        self.list_images =["x"]
        self.setupUi(self)
        self.btnOpen.clicked.connect(self.pegaImagem)
        self.btnNext.clicked.connect(self.configuraSegundaTELA)
        self.btnInfo.clicked.connect(self.abreInfo)
    
    def pegaImagem(self):
        self.contador= "NULL"
        self.contadorNormal = 0
        self.contadorL1 = 0
        self.contadorL2 = 0
        self.contadorL3 = 0
        self.PNormal = 0
        self.P1 = 0
        self.P2 = 0
        self.P3 = 0
        self.PTotal=0
        self.imgAux  = "a"
        self.list_images =["x"]
        fname = QFileDialog.getOpenFileName(self,'Abrir arquivo','c\\','Image files (*.jpg *.gif *.tif *png)')
        path = fname[0]
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(427,529)
        #self.imgI.setPixmap(QPixmap(pixmap))
        img = cv2.imread(path)
        img2 = cv2.imread(path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        kernel2 = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(img,markers)
        idx =0
        classes = ["L1","L2","L3"]
        cont = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        sufix =".png"
        for label in np.unique(markers):
            string = "Celula"
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == label] = 255
            # detecta contorno
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            #desenhar retangulo
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),3)
            #coloca label
            ##print(label)
            label = label -1
            if (label ==0):
                continue;
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # org 
            org = (int(x) - 10, int(y))
            # fontScale 
            fontScale = 1
            # Red color in BGR 
            color = (0, 0, 255) 
            # Line thickness of 2 px 
            thickness = 2
            out = np.zeros_like(img)
            out[mask == 255] = img[mask == 255]
            #Now crop
            (y, x) = np.where(mask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out = out[topy:bottomy+1, topx:bottomx+1]
            dim = (224,224)
            string = string+cont[idx]+sufix
            opening = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel2)
            resized = cv2.resize(out,dim)
            test_image = image.img_to_array(resized)
            test_image = np.expand_dims(test_image, axis = 0)
            res = modelBin.predict_classes(test_image)
            res2="NADA"
            if (res[0][0] == 0):
                res = "ALL"
            else:
                res = "Normal"
                self.contadorNormal = self.contadorNormal + 1
            if (res == "ALL"):
                res2 = [classes[i]for i in modelTipos.predict_classes(test_image)]
                #print (res2)
            #print(res)
            if (res2== "NADA"):
                cv2.putText(img2, res, org, font, fontScale,color, thickness, cv2.LINE_AA, False)   
                cv2.putText(img, res, org, font, fontScale,color, thickness, cv2.LINE_AA, False)   
                #print(res)
                
                self.list_images.insert(idx,[string,res])
            else:
                
                res2 = str(res2);
                res2= res2.replace("[","")
                res2= res2.replace("]","")
                res2= res2.replace(",","")
                res2= res2.replace("'","")
                if (res2== "L1"):
                    self.contadorL1 = self.contadorL1 + 1
                if (res2== "L2"):
                   self.contadorL2 = self.contadorL2 + 1
                if (res2== "L3"):
                   self.contadorL3 = self.contadorL3 + 1
                cv2.putText(img2, res2, org, font, fontScale,color, thickness, cv2.LINE_AA, False)
                cv2.putText(img, res2, org, font, fontScale,color, thickness, cv2.LINE_AA, False)
                #print(res2)
                
                self.list_images.insert(idx,[string,res2])
            cv2.imwrite(string, resized)
            
            self.imgAux = string
            idx = idx +1    
        self.PNormal = self.contadorNormal/idx * 100
        self.P1 = self.contadorL1/idx  * 100
        self.P2 = self.contadorL2/idx * 100
        self.P3 = self.contadorL3/idx  * 100
        self.contador = "Celulas ="+str(idx)
        self.contadorNormal = "Normal: "+str(self.contadorNormal)+" ({0:.2f}".format(self.PNormal)+"%)"
        self.contadorL1 = "L1: "+str(self.contadorL1)+" ({0:.2f}".format(self.P1)+"%)"
        self.contadorL2 = "L2: "+str(self.contadorL2)+" ({0:.2f}".format(self.P2)+"%)"
        self.contadorL3 ="L3: " +str(self.contadorL3)+" ({0:.2f}".format(self.P3)+"%)"
        img2[markers == -1] = [255,255,255]
        cv2.imwrite('result.png',img)   
        pixmap2 = QPixmap('result.png')
        pixmap2 = pixmap2.scaled(859,579)
        self.imgF.setPixmap(pixmap2)
        
        #self.predict(path)
        
    def configuraSegundaTELA(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Form(self.contador,self.imgAux,self.list_images,self.contadorL1,self.contadorL2,self.contadorL3,self.contadorNormal,self.PNormal,self.P1,self.P2,self.P3)
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()
            
    def abreInfo(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Formu()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()
        
 
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        window = MyApp()
        window.show()
        app.exec_()
    run_app()
    k=input("Press any button to exit...") 