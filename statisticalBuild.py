#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:32:30 2019

@author: tang
"""

import numpy as np 
import skimage.io as io
import scipy.io as sio
import keras

import os,random,math

zhuanyiPath = '/home/tang/桌面/SRSSUfilteSamp'

gridpath = zhuanyiPath + '/usefulSamples/GT10grid64ofgtall.mat';
allGrid = sio.loadmat(gridpath)['grid'];    number = len(allGrid);   
sumPix = 0.0;    sumBuild = 0.0; 
allii = []
for ii in range(number):
    OneofshuffleGrid = allGrid[ ii ];  
    imgp = OneofshuffleGrid[0];  GTp = OneofshuffleGrid[1];   
    Xmin = int(OneofshuffleGrid[2]);  Xmax = int(OneofshuffleGrid[3]);
    Ymin = int(OneofshuffleGrid[4]);  Ymax = int(OneofshuffleGrid[5]);
    tempGTp = (zhuanyiPath + GTp).replace(' ','');   
    OneGT = io.imread( tempGTp );
    OneGTuse = OneGT[Xmin:Xmax, Ymin:Ymax];
    size = OneGTuse.shape[0];
    sumPix = sumPix + size*size;
    sumBuild = sumBuild + np.sum( (OneGTuse==255)*1 ) ;
    BULIDPER = 1.0*np.sum( (OneGTuse==255)*1 ) / (size*size)
    if(BULIDPER>0.6):
        allii.append(ii)



percent = sumBuild/sumPix;
