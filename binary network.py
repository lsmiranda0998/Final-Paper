# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:22:11 2019

@author: Leonardo
"""

#Convolutional Neural Network

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras import optimizers
#from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import plot_model
#from tensorflow.python.keras.utils import plot_model
#import cv2





def plot_trainingACC(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
      
        epochs = range(len(acc))
    
        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Acurácia do training_set e do test_set')
        plt.ylabel('acuracia (acc = vermelho, val_acc = azul)')
        plt.xlabel('epoca')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.figure()
        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'b-')
    
        plt.show()
    
    
def plot_trainingLOSS(history):
       # acc = history.history['acc']
        #val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
    
        #plt.plot(epochs, acc, 'r.')
        #plt.plot(epochs, val_acc, 'b')
       # plt.title('Acurácia do training_set e do test_set')
        plt.ylabel('loss (loss = verde, val_loss = blu)')
        plt.xlabel('epoca')
        plt.legend(['train', 'test'], loc='best')
        #plt.figure()
        plt.legend(['train', 'test'], loc='upper left')
        plt.plot(epochs, loss, 'g.')
        plt.plot(epochs, val_loss, 'b-')
        plt.title('Perca do training_set e do test_set')
        plt.show()
    
        plt.savefig('acc_vs_epochs10241024.png')
    
model = Sequential()


model.add(ResNet50(include_top = False,input_shape = (224, 224, 3), pooling = 'max', weights = 'imagenet'))
for layer in model.layers:
    layer.trainable = False
#model.add(Flatten())
model.add(Dense(activation = 'relu', units = 1024))
#model.add(Dropout(0.5))
#model.add(Dropout(0.25))

model.add(Dense(activation = 'sigmoid',units = 1))




model.summary()


#Compilando

adam = optimizers.Adam(1e-3)#1e-5 foi 63%
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])# Categorical cross entropy caso não seja binary

train_datagen = ImageDataGenerator(width_shift_range = 0.2)

test_datagen = ImageDataGenerator(width_shift_range = 0.2)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(224, 224),
                                                batch_size=64,
                                                class_mode='binary',
                                                shuffle = True
                                                #save_to_dir = 'dataset/transformacao/training_set'
                                                )

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(224, 224),
        batch_size=64,
        shuffle = True,
        class_mode='binary')


        #save_to_dir = 'dataset/transformacao/test_set')"""
        
#callback = [EarlyStopping(monitor='val_loss')]
res = model.fit_generator(
       training_set,
       #steps_per_epoch=469,
       epochs=100,
       validation_data=test_set)
       #validation_steps=98)    
       #callbacks = callback)
       
plot_trainingLOSS(res)
plot_trainingACC(res)
#plt.plot(model.history.history['mean_acc'])
   #plt.plot_model(model)
   #plt.show()
model.save_weights('try1024512W.h5')
model.save('try1024512.h5')

#plt.plot(model.history.history['mean_acc'])
#plot_model(model)
#plt.show()

#model.save_weights('try512_256W.h5')
#model.save('try512_256.h5')
#save_model(model,'try512_256T.h5')
"""model = load_model('try1024_1024.h5')
model.summary()

#Y_pred = model.predict_proba(test_set)
print('Confusion Matrix')
print(confusion_matrix(test_set., y_pred))
print('Classification Report')
target_names = list(test_set.class_indices.keys()) 
print(classification_report(test_set.classes,y_pred, target_names=target_names))
Y_pred = model.predict_classes(test_set)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, Y_pred))
print('Classification Report')
target_names = list(test_set.class_indices.keys()) 
print(classification_report(test_set.classes,Y_pred, target_names=target_names))
import sys
from PyQt5 import QtCore, QtWidgets,uic
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget,QPushButton,QFileDialog
from PyQt5.QtCore import QSize   
from PyQt5.QtGui import QPixmap
from keras.preprocessing import image
qtCreatorFile = "telaPrincipal.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.btnOpen.clicked.connect(self.pegaImagem)
    
    def pegaImagem(self):
        fname = QFileDialog.getOpenFileName(self,'Abrir arquivo','c\\','Image files (*.jpg *.gif *.tif *png)')
        path = fname[0]
        pixmap = QPixmap(path)
        self.img.setPixmap(QPixmap(pixmap))
        self.predict(path)
        
    def predict(self,path):
        test_image = image.load_img(path, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict_classes(test_image)
        print(path)
        esperado = path[len(path)-5]
        #training_set.class_indices
        if result[0][0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'ALL'
        if esperado[0] == '0':
            esperado = 'Normal'
        else:
            esperado = 'ALL'
        self.labelResult.setText("Resultado obtido: "+prediction+"\nResultado esperado: "+esperado)
        
 
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        window = MyApp()
        window.show()
        app.exec_()
    run_app()
    
#plt.plot(res.history['loss'])"""