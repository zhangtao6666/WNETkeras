# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:33:07 2020

@author: ZT
"""

import cv2,os,sys
from keras import models
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from skimage import io
import scipy.io

path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));

cor = ListedColormap(['Black', 'Red']);

def changeTO_255(imageArray):
    NB = imageArray.shape[2]
    finallist = []
    for ii in range(NB):
        oneband = imageArray[:,:,ii];
        maxV = np.max(oneband);  minV = np.min(oneband);
        finalBand = 255*(oneband-minV)/(maxV-minV)
        finallist.append(finalBand.astype(np.uint8))
    return np.dstack(finallist)

def changeoneTO_255(imageF):
    oneband = imageF;
    maxV = np.max(oneband);  minV = np.min(oneband);
    finalBand = (255 - 30)*(oneband-minV)/(maxV-minV) + 30;
    return finalBand.astype(np.uint8);


# 将浮点图像转换成有效图像
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


#根据softmax确定最终类别 
def actvationTOclassLabel(predictY):
    oneimg = predictY[0,:,:];
    chazhi = ( (oneimg[:,0] - oneimg[:,1])<0 ) * 255;
    onelabeler = ( np.array(chazhi) ).reshape( 256, 256 );         
    return onelabeler;

#加载模型
WnetModel = load_model( zhuanyiPath + '/saveSTModel/ST_Build64of4bandsUnetmyHRModel.h5');

images= np.load(zhuanyiPath + '/FeaturesVisionIMG/TFbeijing133_3570.npy')
image_arr = np.expand_dims(images, axis=0)

prelabel = actvationTOclassLabel(WnetModel.predict(image_arr, verbose=0));
io.imsave('beijing133_3570Pre.png',prelabel)

# 嵌套在内部，实际可用的模型
modeluseful = WnetModel.get_layer('model_1');
#needLayers=['conv2d_1', 'conv2d_3', 'conv2d_5','conv2d_7','conv2d_9','conv2d_15','conv2d_17','conv2d_21','conv2d_23','conv2d_27','conv2d_31']
needLayers=['conv2d_29']

def plotfeature(Layerfeature,layer_name):
    size = Layerfeature.shape[0];  channel = Layerfeature.shape[2];    margin = 2;
    results = np.zeros((6 * size + 5 * margin, 6 * size + 5 * margin));
    if(channel<=32):
        spaceNumber = 5;
    elif(channel<=64):
        spaceNumber = 10;
    else:
        spaceNumber = 20;
    for i in range(6):  # iterate over the rows of our results grid
        for j in range(6):  # iterate over the columns of our results grid
            oneC_feature = changeoneTO_255(Layerfeature[:,:,i + (j * spaceNumber)]);
            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end] = oneC_feature
    cv2.imwrite(layer_name+'_feature.jpg', results)


def saveSomeFeature(alllayers,trinedModel,image_forPre):
    allFeature={};
    for layer_name in alllayers:
        onelayer = K.function([trinedModel.layers[0].input], [trinedModel.get_layer(layer_name).output]);
        feature = (onelayer([image_forPre])[0])[0,:,:,:]
        allFeature[layer_name] = feature
        plotfeature(feature,layer_name);
    #scipy.io.savemat('WNetallFeature.mat', mdict = allFeature );
    return allFeature;
    
allFeatures = saveSomeFeature(needLayers,modeluseful,image_arr);
#-----------------------------------------------------------#