#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:06:08 2019

@author: tang
"""

import numpy as np 
import skimage.io as io
import scipy.io as sio
import os,sys,math

path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));


#获取文件夹所有图片的路劲和文件名
def file_name(file_dir):
    if (os.path.isdir(file_dir)):
        L=[]; allFilename = [] ;
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                allFilename.append(fileName);
                if (formatName == '.npy') or (formatName == '.png'):
                    tempPath = eval(repr( os.path.join(root, file) ).replace('\\\\', '/'));
                    L.append(tempPath);
        return L,allFilename
    else: print('must be folder path')
#---------------------------------------------------------------------------#

def returnALLXYsizeofIMG(GTfloder):
    allLabelList,allFilename = file_name(GTfloder);
    allXY = [];
    for ii in range(len(allLabelList)):
        oneLabel = io.imread(allLabelList[ii]);
        X = oneLabel.shape[0];   Y = oneLabel.shape[1];
        allXY.append([X,Y]);
    return allXY;



##样本范围框框的生成（有一定的重复率）
def TrainSampleofRepeatScale(Xsize,Ysize, GTfloder, present, reapetScale):
    allLabelList,allFilename = file_name(GTfloder);
    allGrid = [];   allXY = returnALLXYsizeofIMG( GTfloder );
    
    for ii in range( len(allXY) ):
        imagepath =  '/usefulSamples/Image10m_NPY/'+ allFilename[ii] +'.npy' ;
        GtpathST = '/usefulSamples/BuildGT10/' + allFilename[ii] + '.png'
        GTSRpath = '/usefulSamples/BuildGT2p5/' + allFilename[ii] + '.png'

        X = allXY[ii][0];    Y = allXY[ii][1];      jinage = math.floor( Xsize*(1 - reapetScale) );      
        allX0 = np.arange(0,X,jinage);    allY0 = np.arange(0,Y,jinage); 
        
        oneGT = 1*(io.imread( allLabelList[ii]) ==255) ;
        for xii in range( len(allX0) ):
            Xmin = allX0[xii];     Xmax = Xmin + Xsize;
            for yii in range( len(allY0)):
                Ymin = allY0[yii];  Ymax = Ymin + Ysize;
                tempGTsize = oneGT[ Xmin:Xmax, Ymin:Ymax ];
                if( np.sum(tempGTsize) > (Xsize*Ysize)*present ):
                    allGrid.append( ( imagepath,GtpathST, Xmin, Xmax, Ymin, Ymax, GTSRpath ) );

    return allGrid;


def TrainSampleofRepeatScaleLittleBuild(Xsize,Ysize, GTfloder, minS, maxS, reapetScale):
    allLabelList,allFilename = file_name(GTfloder);
    allGrid = [];   allXY = returnALLXYsizeofIMG( GTfloder );
    
    for ii in range( len(allXY) ):
        imagepath =  '/usefulSamples/Image10m_NPY/'+ allFilename[ii] +'.npy' ;
        GtpathST = '/usefulSamples/BuildGT10/' + allFilename[ii] + '.png'
        GTSRpath = '/usefulSamples/BuildGT2p5/' + allFilename[ii] + '.png'

        X = allXY[ii][0];    Y = allXY[ii][1];      jinage = math.floor( Xsize*(1 - reapetScale) );      
        allX0 = np.arange(0,X,jinage);    allY0 = np.arange(0,Y,jinage); 
        
        oneGT = 1*(io.imread( allLabelList[ii]) ==255) ;
        for xii in range( len(allX0) ):
            Xmin = allX0[xii];     Xmax = Xmin + Xsize;
            for yii in range( len(allY0) ):
                Ymin = allY0[yii];  Ymax = Ymin + Ysize;
                tempGTsize = oneGT[ Xmin:Xmax, Ymin:Ymax ];
                if( np.sum(tempGTsize) > (Xsize*Ysize)*minS  and  np.sum(tempGTsize) <= (Xsize*Ysize)*maxS ):
                    allGrid.append( ( imagepath,GtpathST, Xmin, Xmax, Ymin, Ymax, GTSRpath ) );

    return allGrid;


def TrainSampleofRepeatScaleBackGround(Xsize,Ysize,GTfloder, reapetScale):
    allLabelList,allFilename = file_name(GTfloder);
    allGrid = [];   allXY = returnALLXYsizeofIMG( GTfloder );
    
    for ii in range( len(allXY) ):
        imagepath =  '/usefulSamples/Image10m_NPY/'+ allFilename[ii] +'.npy' ;
        GtpathST = '/usefulSamples/BuildGT10/' + allFilename[ii] + '.png'
        GTSRpath = '/usefulSamples/BuildGT2p5/' + allFilename[ii] + '.png'

        X = allXY[ii][0];    Y = allXY[ii][1];      jinage = math.floor( Xsize*(1 - reapetScale) );      
        allX0 = np.arange(0,X,jinage);    allY0 = np.arange(0,Y,jinage); 
        
        oneGT = 1*(io.imread( allLabelList[ii]) ==255) ;
        for xii in range( len(allX0) ):
            Xmin = allX0[xii];     Xmax = Xmin + Xsize;
            for yii in range( len(allY0) ):
                Ymin = allY0[yii];  Ymax = Ymin + Ysize;
                tempGTsize = oneGT[ Xmin:Xmax, Ymin:Ymax ];
                if( np.sum(tempGTsize) == 0 ):
                    allGrid.append( ( imagepath,GtpathST, Xmin, Xmax, Ymin, Ymax, GTSRpath ) );

    return allGrid;


allGrid = TrainSampleofRepeatScale(64,64, zhuanyiPath + '/usefulSamples/BuildGT10', 0.5, 0.0 );

allGridlittle = TrainSampleofRepeatScaleLittleBuild(64,64, zhuanyiPath + '/usefulSamples/BuildGT10', 0.4, 0.5 , 0.0);

# =============================================================================
# import random
# allGridback  = TrainSampleofRepeatScaleBackGround(64,64, zhuanyiPath + '/usefulSamples/BuildGT10', 0.0 )
# 
# randomBack = random.sample(allGridback,0)
# for kk in range(len(randomBack)):
#     oneG = randomBack[kk]
#     allGrid.append(oneG)
# 
# allGridlittle =  TrainSampleofRepeatScaleLittleBuild(64,64, zhuanyiPath + '/usefulSamples/BuildGT10',0.0, 0.1 ,0.0)
# 
# randomLittle = random.sample(allGridlittle,400)
# for mm in range(len(randomLittle)):
#     little = randomLittle[mm]
#     allGrid.append(little)
# =============================================================================

#sio.savemat( zhuanyiPath + '/usefulSamples/GT10grid64ofgtall.mat', {'grid':allGrid});
