#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 01:35:42 2018

@author: caoxiya
"""

from __future__ import division
import os
import numpy as np
import time
import PIL.Image as pil
import tensorflow as tf
from SfMLearner import SfMLearner

def predictDepth(filelist):
    img_height=128
    img_width=416
    ckpt_file = 'models/model-190532'
    sfm = SfMLearner()
    sfm.setup_inference(img_height,
                        img_width,
                        mode='depth')
    depth = []
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        for filename in filelist:
            fh = open(filename, 'r')
            I = pil.open(fh)
            I = I.resize((img_width, img_height), pil.ANTIALIAS)
            I = np.array(I)
            pred = sfm.inference(I[None,:,:,:], sess, mode='depth')
            depth.append(pred['depth'][0,:,:,0])
    depth = np.array(depth)
    np.save("output/depth.npy", depth)
    return depth

depth = predictDepth(['pic/sample.png', 'pic/sample1.png'])