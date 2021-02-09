# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:46:59 2019

@author: zhang
"""
from keras.callbacks import LearningRateScheduler
from UnetSR import modelUnetHR, ReadTrainDataSR, ReadTrainDataSRfromNPY
from keras.callbacks import ModelCheckpoint
import DrawLossAndAccChart as DLAC
from keras.utils import multi_gpu_model
import scipy.io as sio
from keras.optimizers import Adam,SGD
import os,sys,math
import tensorflow as tf


#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));


activation = 'softmax';      batchS = 32;        Nbands = 4;     size = 64;     useGPU = True; 

pretrained_weights = zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnetmyHRweight.h5';
gridpath = zhuanyiPath + '/usefulSamples/GT10grid64ofgtall.mat';


mytrainGene = ReadTrainDataSR.trainSampleGenerator(activation, batchS, size, gridpath, shuffle = True);
myValidate = ReadTrainDataSR.ValSample(activation, 20, size, gridpath);


with tf.device('/cpu:0'):
    UnetModel = modelUnetHR.UnetmyHR(activation, size, size, Nbands);

if (useGPU==True):
    UnetModel =  multi_gpu_model(UnetModel, gpus= 8); #将搭建的model复制到4个GPU中

if (os.path.exists(pretrained_weights)):
    UnetModel.load_weights(pretrained_weights,by_name=True)
    UnetModel.trainable = True

initial_lrate = 0.01;          save_weights = True;
# epoch是10的整数倍时，学习率重置初始值，连续两个整十数之间呈指数下降
def Exponentialdecay(epoch):
    drop = 0.9;
    yushu = epoch % 100 ;
    if(yushu == 0):
        lrate = initial_lrate;
    else:
        lrate = initial_lrate*math.pow(drop,yushu);
    return lrate;

mylrate = LearningRateScheduler(Exponentialdecay);


optimal = SGD(lr = initial_lrate, momentum = 0.9, decay=0.0, nesterov=False);  # Adam(lr = 0.002)
if(activation == 'sigmoid'):
    UnetModel.compile(optimizer = optimal, loss = 'binary_crossentropy', metrics = ['accuracy'] )  # sgd,rmsprop   ['accuracy']
else:
    UnetModel.compile(optimizer = optimal, loss = 'categorical_crossentropy', metrics = ['accuracy'] ) 


model_checkpoint = ModelCheckpoint( pretrained_weights, monitor='acc',verbose=1, save_best_only=True, save_weights_only = save_weights)
history = DLAC.LossHistory();


myHis = UnetModel.fit_generator(mytrainGene, steps_per_epoch = 250,epochs = 50, 
                         callbacks = [mylrate, model_checkpoint,history], 
                         validation_data = myValidate )

UnetModel.save(zhuanyiPath +  '/saveSTModel/WNetModel.h5');
#history.loss_plotMytime('epoch');

allacc = myHis.history['acc'];   allloss = myHis.history['loss'];
valacc = myHis.history['val_acc'];  valloss = myHis.history['val_loss'];

fw = open(zhuanyiPath +  '/saveSTModel/WNetModel.txt', 'a', encoding = 'utf-8' )
for index in range(len(allacc)):
    onehang = 'acc: '+ str(allacc[index]) + ',' + 'loss: ' + str(allloss[index]) + ',' + 'val_acc: ' + str(valacc[index]) + ',' + 'val_loss: ' + str(valloss[index]) + '\n'
    fw.write(onehang)
fw.close();
