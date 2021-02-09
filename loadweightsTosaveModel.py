# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:07:34 2020

@author: ZT
"""

from UnetSR import modelUnetHR
from  Unetcode import  modelUNET
from keras.utils import multi_gpu_model
import os,sys
import tensorflow as tf
from keras.optimizers import SGD

path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));
optimal = SGD(lr = 0.01, momentum = 0.9, decay=0.0, nesterov=False);  


WNet_weights = zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnetmyHRweight.h5';
with tf.device('/cpu:0'):
    WnetModelK = modelUnetHR.UnetmyHR('softmax', 64, 64, 4);
useGPU = True;
if (useGPU==True):
    WnetModel =  multi_gpu_model(WnetModelK, gpus= 8); #将搭建的model复制到4个GPU中

if (os.path.exists(WNet_weights)):
    WnetModel.load_weights(WNet_weights,by_name=True); 
    WnetModel.trainable = True;  
    WnetModel.compile(optimizer = optimal, loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    WnetModel.save(zhuanyiPath +  '/saveSTModel/WNetModel.h5');  print(WnetModel.summary());


UNet10_weights = zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnet10weight.h5';
with tf.device('/cpu:0'):
    Unet10ModelK = modelUNET.Unetmy('softmax', 64, 64, 4);
if (useGPU==True):
    Unet10Model = multi_gpu_model(Unet10ModelK, gpus= 8); #将搭建的model复制到4个GPU中

if (os.path.exists(UNet10_weights)):
    Unet10Model.load_weights(UNet10_weights,by_name=True)
    Unet10Model.trainable = True;  
    Unet10Model.compile(optimizer = optimal, loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    Unet10Model.save(zhuanyiPath +'/saveSTModel/UNet10Model.h5');  print(Unet10Model.summary());



