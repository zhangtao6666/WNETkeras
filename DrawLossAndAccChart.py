# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:59:56 2018

@author: zhang
"""
import keras
import matplotlib.pyplot as plt
import time
import numpy as np

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        self.mytime = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        self.mytime['batch'].append(time.time())

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.mytime['epoch'].append(time.time());

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(figsize=(20,15))
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type,fontsize=25);  plt.xticks(fontsize=20);   
        plt.ylabel('acc-loss',fontsize=25); 
        plt.ylim(0, 1.2); plt.yticks(fontsize=20);
        plt.legend(loc="upper right",ncol = 2,fontsize = 24);
        plt.show()
        
        
    def loss_plotMytime(self, loss_type):
        #iters = self.mytime[loss_type] - self.mytime[loss_type][0];
        iters = np.array(self.mytime[loss_type]);
        myiters = iters -iters[0];
        allepochs = range(len(self.losses[loss_type]))
        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(111)
        # acc
        ax1.plot(allepochs, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        ax1.plot(allepochs, self.losses[loss_type], 'g', label='train loss')
        ax1.lines.pop(0);   ax1.lines.pop(0);
        
        ax2 = plt.twiny();   # 顶上X轴为时间信息
        # acc
        ax2.plot(myiters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        ax2.plot(myiters, self.losses[loss_type], 'g', label='train loss')
        
        if loss_type == 'epoch':
            # val_acc
            ax1.plot(allepochs, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            ax1.plot(allepochs, self.val_loss[loss_type], 'k', label='val loss')
            ax1.lines.pop(0);  ax1.lines.pop(0);
            
            # val_acc
            ax2.plot(myiters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            ax2.plot(myiters, self.val_loss[loss_type], 'k', label='val loss')
             

        ax1.set_xlabel(loss_type,fontsize=25);     ax2.set_xlabel('time(s)',fontsize=25);    
        ax2.legend(loc="upper right",ncol = 2,fontsize = 24); ax1.set_ylabel('acc-loss',fontsize=25);       
        ax1.locator_params("x", nbins = 20);  ax2.locator_params("x", nbins = 20);
          
        ax1.grid(True);
        
        ax = plt.gca();     ax.set_ylim(0, 1.2);   
        plt.setp(ax1.get_xticklabels(), fontsize=20);  plt.setp(ax2.get_xticklabels(), fontsize=20);
        plt.setp(ax1.get_yticklabels(), fontsize=20);
        plt.savefig('trainprocess.png');
        plt.show();
        
import keras.backend as K

#from keras import losses
def myLosscrossentropy(y_truetensor, y_predtensor, WT = 0.001):
    #--------------mask之内交叉熵损失——————————————————-------#
    inMask = K.cast(K.not_equal(y_truetensor, 3.0), K.floatx());   
    y_trueinMask = inMask*y_truetensor;       y_predInmask = inMask*y_predtensor;
    #crossentropy = K.mean(K.square(y_predInmask - y_trueinMask), axis=-1); 
    crossentropy = K.categorical_crossentropy(y_trueinMask,y_predtensor);
    #-----------------------------------------------------------#
    #--------------mask之外 -log(abs(y_predtensor[:,:,0]-y_predtensor[:,:,1]))———-------#
    outMask = K.cast(K.equal(y_truetensor, 3.0), K.floatx());
    quanwei1 =  K.cast(K.not_equal(y_predtensor, 255.0), K.floatx());
    y_predfan = quanwei1 - y_predtensor;
    y_predOutmask = outMask*y_predtensor;     y_predOutmaskFan = outMask*y_predfan; 
    diffenceLoss = K.mean( K.tf.pow( 0.001 ,K.abs(y_predOutmask - y_predOutmaskFan) ), axis=-1);
    #------------------------------------------------------------------------------------#
    return (1-WT)*crossentropy + WT*diffenceLoss ; # 


def iou(y_true, y_pred, label: int):
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)
        
def mean_iou(y_true, y_pred):
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)    
    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels


def Precision(y_true, y_pred):
    """精确率"""
    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1)));  # true positives
    pp= K.sum(K.round(K.clip(y_pred, 0, 1))); # predicted positives
    precision = tp/ (pp+ K.epsilon());
    return precision;
    
def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))); # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1))); # possible positives
    recall = tp / (pp + K.epsilon());
    return recall;
 
def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred);
    recall = Recall(y_true, y_pred);
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()));
    return f1;


# PredictLabel和TrueLabel都是二值化，1代表目标，0代表背景

def CalThreeIndex(PredictLabel,TrueLabelBinary):
    #计算PA （预测正确的像素占总像素的比例）
    TrueLabel = TrueLabelBinary*1.0;
    TrueLabelPT = (PredictLabel == TrueLabel) + 0;
    numberTrue = np.sum(TrueLabelPT);
    NumberALL = PredictLabel.shape[0] * PredictLabel.shape[1];
    PA = (1.0*numberTrue) / (NumberALL*1.0);
    #计算Precision  精确率 
    # Precision​ Ratio: 查准率，真正是正类的像素个数 占 预测出的是正类的比例。    
    intersect = ( (PredictLabel + TrueLabel) == 4 ) + 0;
    numINtersect = np.sum(intersect);
    predictBuild = (PredictLabel==2) + 0;
    numberPRE = np.sum(predictBuild);
    precision =  (1.0*numINtersect) / (1.0*numberPRE);
    #计算 recall  
    # Recall Ratio: 查全率，真正是正类的像素个数，占整个Ground Truth中真正正类个数的比例。
    GTbuild =  (TrueLabel == 2) + 0 ;
    numberGTbuild = np.sum(GTbuild);
    recall =  (1.0*numINtersect) / (1.0*numberGTbuild);
    #计算IOU（交并比）
    union = ( (PredictLabel + TrueLabel) >= 3 ) + 0;
    numunion = np.sum(union);
    IOU = (1.0*numINtersect) / (1.0*numunion);
    #final = [IOU, PA, precision, recall];
    return IOU, PA, precision, recall;


def CalThreeIndexFORBuild(PredictLabel,TrueLabelBinary):
    PredictLabel = (PredictLabel==255)*10
    sumV = PredictLabel + TrueLabelBinary
    intersect = (sumV==12) + 0
    intersectnum = np.sum( intersect );
    #计算IOU（交并比）
    union = (sumV>=2) + 0 
    unionnum = np.sum(union);
    IOU = (1.0*intersectnum) / (1.0*unionnum);
    #计算Precision  精确率 
    # Precision​ Ratio: 查准率，真正是正类的像素个数 占 预测出的是正类的比例。    
    predictBuild = (PredictLabel==10) + 0;
    numberPRE = np.sum(predictBuild);
    precision =  (1.0*intersectnum) / (1.0*numberPRE);
    #计算 recall  
    # Recall Ratio: 查全率，真正是正类的像素个数，占整个Ground Truth中真正正类个数的比例。
    GTbuild =  (TrueLabelBinary == 2) + 0 ;
    numberGTbuild = np.sum(GTbuild);
    recall =  (1.0*intersectnum) / (1.0*numberGTbuild);

    return IOU, precision, recall;


