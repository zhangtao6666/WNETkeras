# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:51:59 2020

@author: ZT
"""
import os,sys,cv2
from keras import backend as K
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from keras.models import load_model


path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));

#加载模型
WnetModel = load_model( zhuanyiPath + '/saveSTModel/ST_Build64of4bandsUnetmyHRModel.h5');

#测试影像
images= np.load(zhuanyiPath + '/FeaturesVisionIMG/TFbeijing133_3570.npy')
image_arr = np.expand_dims(images, axis=0)

# 嵌套在内部，实际可用的模型
modeluseful = WnetModel.get_layer('model_1');

#------------------卷积核-convnet filters------------------------#
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

def generate_pattern(model, layer_name, filter_index, size=64):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.layers[0].input)[0]
    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # This function returns the loss and grads given the input picture
    iterate = K.function([model.layers[0].input], [loss, grads])
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 4));
    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

#plt.imshow(generate_pattern(modeluseful,'conv2d_2', 0));  plt.show();


needLayers=['conv2d_1', 'conv2d_3', 'conv2d_5','conv2d_7','conv2d_9','conv2d_15','conv2d_17','conv2d_21','conv2d_23','conv2d_27','conv2d_31']
lastlayer = ['conv2d_29'];
for layer_name in lastlayer:
    size = 64;    margin = 3;
    if(layer_name=='conv2d_1'):
        spaceNumber = 10;
    elif(layer_name=='conv2d_31'):
        spaceNumber = 5;
    else:
        spaceNumber = 20;
    # This a empty (black) image where we will store our results.
    results = np.zeros((6 * size + 5 * margin, 6 * size + 5 * margin, 3))
    for i in range(6):  # iterate over the rows of our results grid
        for j in range(6):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 间隔space)` in `layer_name`
            filter_img = generate_pattern(modeluseful,layer_name, i + (j * spaceNumber), size=size)
            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img[:,:,0:3]
    # Display the results grid
    cv2.imwrite(layer_name+'_filter.jpg', results)
    #plt.figure(figsize=(25, 25))
    #plt.imshow(results)
    #plt.show()