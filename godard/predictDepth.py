from __future__ import division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import cv2
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


def convert_disps_to_depths_kitti(disparities, height, width):
    pred_depths = []
    for i in range(len(disparities)):
        pred_disp = disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
        pred_depth = width_to_focal[width] * 0.54 / pred_disp
        pred_depths.append(pred_depth)
    return pred_depths

def predictDepth(filelist, pwd):

    encoder = 'vgg'
    input_height = 256
    input_width = 512
    disparities = []
    params = monodepth_parameters(
        encoder,
        input_height,
        input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    left = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = pwd + '/godard/models/model_city2kitti'
    train_saver.restore(sess, restore_path)

    for image_path in filelist:
        input_image = scipy.misc.imread(image_path, mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        print(original_height, original_width)
        input_image = scipy.misc.imresize(input_image, [input_height, input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
        disparities.append(disp[0].squeeze())

    depth = convert_disps_to_depths_kitti(disparities, original_height, original_width)
    depth = np.asarray(depth)
    np.save(pwd+"/godard/output/depth.npy", depth)
    return depth

#filelist = ['/Users/caoxiya/Desktop/monodepth/data/0000000000.png',
#            '/Users/caoxiya/Desktop/monodepth/data/0000000001.png',
#            '/Users/caoxiya/Desktop/monodepth/data/0000000002.png']
#model_path = '/Users/caoxiya/Desktop/monodepth/models/model_city2kitti'
#predictDepth(filelist, model_path)