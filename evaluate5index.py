# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:25:09 2019

@author: zhang
"""
from sklearn.metrics import confusion_matrix
import skimage.io as io
from PIL import Image
import os
# 需要先设置内存限制，不然仍然会报错内存溢出
Image.MAX_IMAGE_PIXELS = None


#获取文件夹所有图片的路劲和文件名
def file_name(file_dir):
    if (os.path.isdir(file_dir)):
        L=[]; allFilename = [] ;
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                allFilename.append(fileName);
                if (formatName == '.tif'):
                    tempPath = eval(repr( os.path.join(root, file) ).replace('\\\\', '/'));
                    L.append(tempPath);
        return L,allFilename
    else: print('must be folder path')
#---------------------------------------------------------------------------#

Unet10fd = '/home/tang/桌面/SRSSUfilteSamp/predictLabel/trainUnet' 
GT10FD = '/home/tang/桌面/UNETbuilding/data10/Build10GeoTIF'

SRfd = '/home/tang/桌面/SRSSUfilteSamp/predictLabel/trainSR' 
GTSR =  '/home/tang/桌面/UNETbuilding/data10/Build2p5GeoTIF'


def calculateAcc(gtpath,resultpath):
    matGT = io.imread(gtpath);
    matRE = io.imread(resultpath);
    matGTuse = matGT[:,:] -1 ;         matREuse = (matRE[:,:] == 255) + 0;
    matGTfinal = matGTuse.flatten();   matREfinal = matREuse.flatten(); 
    
    cm = confusion_matrix(matGTfinal, matREfinal)
    tp = 1.0*cm[1,1];  fn = 1.0*cm[1,0];  fp = 1.0*cm[0,1];   tn = 1.0*cm[0,0];
    oa = (tp+tn)/(tp+tn+fn+fp);
    recall = tp/(tp+fn);
    precision = tp/(tp+fp);
    f1 = 2*precision*recall/(precision+recall)
    iou = tp/(tp+fn+fp);
    return oa,recall,precision,f1,iou;


def batchCA(gtfd,prfd,txtname):
    listfile,allFilename = file_name(gtfd);
    lenth = len(listfile)
    for ii in range(lenth):
        gtpath = listfile[ii]
        resultpath = prfd + '/' + allFilename[ii] +'.tif'
        oa,recall,precision,f1,iou = calculateAcc(gtpath,resultpath)
        fw = open('/home/tang/桌面/SRSSUfilteSamp/' + txtname, 'a', encoding = 'utf-8' )
        onehang = allFilename[ii]+ ',' + str(oa) + ',' + str(recall) + ',' + str(precision)  + ',' + str(f1) + ',' + str(iou) + '\n'
        fw.write(onehang)
        fw.close();

        
batchCA(GT10FD,Unet10fd,'trainacc10.txt');        
batchCA(GTSR,SRfd,'trainacc2p5.txt');         