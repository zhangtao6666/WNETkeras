# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:41:17 2019

@author: zhang
"""

import numpy as np 
import os,math,gdal
import skimage.io as io
from keras.models import load_model
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));


def SaveMATGeoTIFF(panPath,TIFFname,finalGT):
    resultTIFF = finalGT;
    In_ds = gdal.Open( panPath );
    gtiff_driver = gdal.GetDriverByName('GTiff');
    Out_Ds = gtiff_driver.Create( TIFFname, resultTIFF.shape[1] , resultTIFF.shape[0] , 1,  gdal.GDT_Byte  );
    Out_Ds.SetProjection(In_ds.GetProjection() );
    Out_Ds.SetGeoTransform ( In_ds.GetGeoTransform() );
    out_Band = Out_Ds.GetRasterBand(1);   
    out_Band.WriteArray( resultTIFF);
    del Out_Ds;


def SaveSRMATGeoTIFF(panPath,TIFFname,MAT):
    resultTIFF = MAT;
    In_ds = gdal.Open( panPath );
    gtiff_driver = gdal.GetDriverByName('GTiff');
    In_dsT = In_ds.GetGeoTransform() 
    INSRGT = gdal.Open('G:/SRSSNetKeras/ROCaccuracyEva/TFnanjingGT25.tif');
    SRT = INSRGT.GetGeoTransform()
    myTransform = (In_dsT[0], SRT[1], In_dsT[2], In_dsT[3], In_dsT[4], SRT[5])
    Out_Ds = gtiff_driver.Create( TIFFname, resultTIFF.shape[1] , resultTIFF.shape[0] , 1,  gdal.GDT_Byte  );
    Out_Ds.SetProjection(In_ds.GetProjection() );
    Out_Ds.SetGeoTransform(myTransform);
    out_Band = Out_Ds.GetRasterBand(1);   
    out_Band.WriteArray( resultTIFF);
    del Out_Ds;


# 将影像size扩展至Xs,Ys的倍数
def OneImgExtent(ImgPath,Xs,Ys):
    oneIMG = io.imread(ImgPath);    X = oneIMG.shape[0];    
    Y = oneIMG.shape[1];            NB = oneIMG.shape[2]; 
    NumX = math.floor(X/Xs) + 1;      NumY = math.floor(Y/Ys) + 1;
    imageExtent = np.zeros((NumX*Xs,NumY*Ys,NB),dtype = np.float);
    imageExtent[0:X,0:Y,:] = oneIMG;
    return imageExtent;

# 从一个影像中生成多个固定尺寸的patch
def MakePatchIMGFromeOneImage(ImgPath,Xs,Ys):
    oneIMG = OneImgExtent(ImgPath,Xs,Ys);   
    X = oneIMG.shape[0];           Y = oneIMG.shape[1]; 
    NumX = math.floor(X/Xs);      NumY = math.floor(Y/Ys);
    allPatchIMG = [];
    for ii in range(NumX):
        for jj in range(NumY):
            OnepatchIMG = oneIMG[ ii*Xs:(ii+1)*Xs , jj*Ys:(jj+1)*Ys ,:];
            allPatchIMG.append( OnepatchIMG ); 
    batchIMg = np.array(allPatchIMG);
    return  batchIMg; 

#根据sigmoid函数或者softmax确定最终类别 
def actvationTOclassLabel(predictY,actvation, Xs, Ys,SR):
    Number = predictY.shape[0];
    allLabelY = [];
    for ii in range(Number):
        oneimg = predictY[ii,:,:];
        X = oneimg.shape[0];   Y = oneimg.shape[1];
        oneimguse = oneimg.reshape(X, Y);
        if (actvation == 'sigmoid'):
            oneLabel = np.where(oneimguse > 0.5, 255, 0)
            allLabelY.append(oneLabel);
        else:
            chazhi = ( (oneimg[:,0] - oneimg[:,1])<0 ) * 255;
            onelabeler = ( np.array(chazhi) ).reshape( Xs*SR, Ys*SR );
            allLabelY.append(onelabeler);         
    return np.array(allLabelY);
#---------------------------------------------------------------------------#

#-----利用训练好的CNN对待分类数据预测Label------没有groundtruth-----------#
def PredictTestdataNoGT(testIMGpath, Trainedmodel, Xs,Ys, savename, actvation, SR):
    print(testIMGpath)
    OneIMG = io.imread(testIMGpath);      Xsize = OneIMG.shape[0];     Ysize = OneIMG.shape[1]; 
    NumX = math.floor(Xsize/Xs)+1;   NumY = math.floor(Ysize/Ys)+1;
    batchIMg = MakePatchIMGFromeOneImage(testIMGpath,Xs,Ys);
    predictY = Trainedmodel.predict( batchIMg, batch_size=16, verbose=1);
    allLabelY = actvationTOclassLabel(predictY,actvation, Xs, Ys, SR );
    allHang = []; 
    for jj in range(NumX):
        oneHnage = [];
        for kk in range(NumY):
            temp = allLabelY[ jj*NumY + kk, :, :];
            oneHnage.append( temp );
        finalLabelOnehang = np.hstack( (oneHnage) );
        allHang.append( np.array( finalLabelOnehang ) );
    finalLabelPre = np.vstack( (allHang) );
    finalLabel = finalLabelPre[0:Xsize*SR,0:Ysize*SR]; 
    savepath = os.path.join('G:/SRSSNetKeras/ROCaccuracyEva/predictLabel/'+ savename +'.tif');
    if(SR>1):
        SaveSRMATGeoTIFF(testIMGpath,savepath,finalLabel);
    else:
        SaveMATGeoTIFF(testIMGpath,savepath,finalLabel);
    return finalLabel;
#---------------------------------------------------------------------------# 

#获取文件夹所有图片的路劲和文件名
def file_name(file_dir):
    if (os.path.isdir(file_dir)):
        L=[]; allFilename = [] ;
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                if (formatName == '.tif' and formatName != '.tif.aux.xml'):
                    allFilename.append(fileName);
                    tempPath = eval(repr( os.path.join(root, file) ).replace('\\\\', '/'));
                    L.append(tempPath);
        return L,allFilename
    else: print('must be folder path')
#---------------------------------------------------------------------------#


isSR = False;
Listfile,allFilename = file_name('G:/SRSSNetKeras/ROCaccuracyEva/IMG10m');
for ii in range(len(Listfile)):
    if(isSR):
        savename = 'WNet/' + allFilename[ii];
        print(savename);
        unetModel = load_model( zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnetmyHRModel.h5');
        FinalLabel =  PredictTestdataNoGT( Listfile[ii], unetModel, 64, 64, savename, 'softmax', 4); 
    else:
        savename = 'UNet10/' + allFilename[ii];
        print(savename);
        unetModel = load_model( zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnet10model.h5');
        FinalLabel =  PredictTestdataNoGT( Listfile[ii], unetModel, 64, 64, savename, 'softmax', 1);  
