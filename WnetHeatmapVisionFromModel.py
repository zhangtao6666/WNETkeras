# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:46:38 2020

@author: ZT
"""

import os,sys
from keras import backend as K
import numpy as np
from keras.models import load_model
import cv2
import scipy.ndimage as SNDIG

path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));

#加载模型
WnetModel = load_model( zhuanyiPath + '/saveSTModel/ST_Build64of4bandsUnetmyHRModel.h5');

#测试影像
images= np.load(zhuanyiPath + '/FeaturesVisionIMG/TFbeijing133_3570.npy')
image_arr = np.expand_dims(images, axis=0)

# 嵌套在内部，实际可用的模型
modeluseful = WnetModel.get_layer('model_1');
print(modeluseful.summary());


def output_heatmap(model, last_conv_layerN, img):
    """Get the heatmap for image.
    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model. string
           img: processed input image.
    Returns:
           heatmap: heatmap.
    """
    # This is the entry in the prediction vector
    target_output = model.layers[-1].output;

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layerN)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads 
    # given the input picture
    iterate = K.function([model.layers[0].input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
    #for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def Upsample(imageArray):
    NB = imageArray.shape[2]
    finallist = []
    for ii in range(NB):
        oneband = imageArray[:,:,ii];
        finalBand = SNDIG.zoom(oneband, 4, order=0);
        finallist.append(finalBand.astype(np.uint8))
    return np.dstack(finallist)

def changeoneTO_255(imageF):
    oneband = imageF;
    maxV = np.max(oneband);  minV = np.min(oneband);
    finalBand = (255 - 0)*(oneband-minV)/(maxV-minV);
    return finalBand.astype(np.uint8);

# use cv2 to load the original image
imgfirst = cv2.imread(zhuanyiPath + '/FeaturesVisionIMG/TFbeijing133_3570RGB.png')
img = Upsample(imgfirst);

# convert the heatmap to RGB
heatmap = output_heatmap(modeluseful, 'conv2d_34', image_arr)
heatmap = changeoneTO_255(heatmap)
cv2.imwrite('beijing133_3570_heatmapBack.jpg', heatmap);
cv2.imwrite('beijing133_3570_heatmapBuild.jpg', 255 - heatmap);

# 0.4 here is a heatmap intensity factor
#superimposed_img = heatmap * 0.4 + img;
# Save the image to disk
# cv2.imwrite('beijing133_3570_heatmap.jpg', superimposed_img);