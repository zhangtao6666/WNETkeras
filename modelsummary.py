# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:11:36 2020

@author: ZT
"""
import os,sys
from keras.models import load_model
from keras.utils import plot_model

path = os.path.abspath(os.path.dirname(sys.argv[0]))
zhuanyiPath = eval(repr( path ).replace('\\\\', '/'));

WNet = zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnetmyHRModel.h5';
WNetmodel = load_model(WNet)
plot_model(WNetmodel, to_file='WNetmodel.png', show_shapes='True')
print(WNetmodel.summary())

UNet10 = zhuanyiPath +  '/saveSTModel/ST_Build64of4bandsUnet10model.h5';
UNet10model = load_model(UNet10)
plot_model(UNet10model, to_file='UNet10model.png', show_shapes='True')
print(UNet10model.summary())