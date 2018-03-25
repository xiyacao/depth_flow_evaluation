'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import time
import numpy as np
import ipdb

import theano
import theano.tensor as T

from common import imgutil, logutil

import matplotlib.pyplot as plt

import thutil
from thutil import test_shape, theano_function, maximum

from net import *
from pooling import cmrnorm

_log = logutil.getLogger()
xx = np.newaxis

def _image_montage(imgs, min, max):
    imgs = imgutil.bxyc_from_bcxy(imgs)
    return imgutil.montage(
        imgutil.scale_values(imgs, min=min, max=max),
        border=1)

def _depth_montage(depths):
    if depths.ndim == 4:
        assert depths.shape[1] == 1
        depths = depths[:,0,:,:]
    #depths = imgutil.scale_values(depths, min=-2.5, max=2.5)
    #depths = map(imgutil.scale_values, depths)
    masks = []
    for i in xrange(len(depths)):
        x = depths[i]
        mask = x != x.min() 
        masks.append(mask)
        x = x[mask]
        if len(x) == 0:
            d = np.zeros_like(depths[i])
        else:
            d = imgutil.scale_values(depths[i], min=x.min(), max=x.max())
        depths[i] = d
    depths = plt.cm.jet(depths)[...,:3]
    for i in xrange(len(depths)):
        for c in xrange(3):
            depths[i, :, :, c][masks[i] == 0] = 0.2
    return imgutil.montage(depths, border=1)

def _zero_pad_batch(batch, bsize):
    assert len(batch) <= bsize
    if len(batch) == bsize:
        return batch
    n = batch.shape[0]
    shp = batch.shape[1:]
    return np.concatenate((batch, np.zeros((bsize - n,) + shp,
                                           dtype=batch.dtype)))

class machine(Machine):
    def __init__(self, conf):
        Machine.__init__(self, conf)

    def infer_depth(self, images):
        '''
        Infers depth maps for a list of 320x240 images.
        images is a nimgs x 240 x 320 x 3 numpy uint8 array.
        returns depths (nimgs x 55 x 74) corresponding to the center box
        in the original rgb image.
        '''
        images = images.transpose((0,3,1,2))
        (nimgs, nc, nh, nw) = images.shape
        assert nc == 3 and (nh, nw) == self.orig_input_size

        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size

        bsize = self.bsize
        b = 0

        # theano function for inference
        v = self.vars
        pred_depth = self.inverse_depth_transform(self.fine.pred_mean)
        infer_f = theano.function([v.images], pred_depth)

        depths = np.zeros((nimgs, output_h, output_w), dtype=np.float32)

        # crop region (from random translations in training)
        dh = nh - input_h
        dw = nw - input_w
        (i0, i1) = (dh/2, nh - dh/2)
        (j0, j1) = (dw/2, nw - dw/2)

        # infer depth for images in batches
        b = 0
        while b < nimgs:
            batch = images[b:b+bsize]
            n = len(batch)
            if n < bsize:
                batch = _zero_pad_batch(batch, bsize)

            # crop to network input size
            batch = batch[:, :, i0:i1, j0:j1]

            # infer depth with nnet
            depths[b:b+n] = infer_f(batch)[:n]
            
            b += n

        return depths

    def inverse_depth_transform(self, logdepths):
        # map network output log depths back to depth
        # output bias is init'd with the mean, and output is logdepth / stdev
        return T.exp(logdepths * self.meta.logdepths_std)

    def get_predicted_depth_region(self):
        '''
        Returns the region of the original image covered by the predicted
        depth map (y0 y1 x0 x1)
        '''
        # original kitti res
        ycrop = 130
        (y0, y1, x0, x1) =  (4+ycrop, 220+ycrop, 44, 1180)
        # downsampled by 2: relative to (184, 612) input image
        return (y0/2, y1/2, x0/2, x1/2)

    def define_machine(self):
        self.orig_input_size = (184, 612) # before data transforms
        self.input_size = (172, 576) # after data transforms

        # size ended up slightly changing during development; forgot to
        # set coarse config back to final shape so its output is a bit larger
        self.coarse_output_size = self.conf.geteval('full2', 'output_size') # 28,144
        self.output_size = self.conf.geteval('data', 'output_size') # 27,142

        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size

        self.define_meta()

        # input vars
        images = T.tensor4('images')
        depths = T.tensor3('depths')
        masks = T.tensor3('masks')

        test_values = self.make_test_values()
        images.tag.test_value = test_values['images']
        depths.tag.test_value = test_values['depths']
        masks.tag.test_value  = test_values['masks']
       
        x0 = images
        # features for coarse stack from imagenet
        self.define_imagenet_stack(x0)

        # pretrained features are rather large, rescale down to nicer range
        imnet_r5 = 0.01 * self.imagenet.r5
        imnet_feats = imnet_r5.reshape((
                            self.bsize, T.prod(imnet_r5.shape[1:])))

        # rest of coarse stack
        self.define_coarse_stack(imnet_feats)

        # fine stack
        self.define_fine_stack(x0)

        self.vars = MachinePart(locals())

    def define_meta(self):
        '''
        precomputed means and stdev
        '''
        # just hardcoding for this release, was in meta.mat file
        images_mean = 95.42509484
        images_std = 79.23276058
        images_istd = 1.0 / images_std
        depths_mean = 15.47680156
        depths_std = 11.80950493
        depths_istd = 1.0 / depths_std
        logdepths_mean = 2.54185751
        logdepths_std = 0.58423747
        logdepths_istd = 1.0 / logdepths_std
        self.meta = MachinePart(locals())

    def make_test_values(self):
        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size
        test_images_size = (self.bsize, 3, input_h, input_w)
        test_depths_size = (self.bsize, output_h, output_w)

        test_values = {}
        test_values['images'] = \
            (255 * np.random.rand(*test_images_size)).astype(np.float32)
        test_values['depths'] = \
            np.random.randn(*test_depths_size).astype(np.float32)
        test_values['masks'] = \
            np.ones(test_depths_size, dtype=np.float32)
        return test_values

    def define_imagenet_stack(self, x0):
        conv1 = self.create_unit('imnet_conv1')
        pool1 = self.create_unit('imnet_pool1')
        conv2 = self.create_unit('imnet_conv2')
        pool2 = self.create_unit('imnet_pool2')
        conv3 = self.create_unit('imnet_conv3')
        conv4 = self.create_unit('imnet_conv4')
        conv5 = self.create_unit('imnet_conv5')
        pool5 = self.create_unit('imnet_pool5')

        z1 = conv1.infer(x0 - 128)
        (p1, s1) = pool1.infer(z1)
        r1 = cmrnorm(relu(p1))

        z2 = conv2.infer(r1)
        (p2, s2) = pool2.infer(z2)
        r2 = cmrnorm(relu(p2))

        z3 = conv3.infer(r2)
        r3 = relu(z3)

        z4 = conv4.infer(r3)
        r4 = relu(z4)

        z5 = conv5.infer(r4)
        (p5, s5) = pool5.infer(z5)
        r5 = relu(p5)

        #r5_vec = r5.reshape((r5.shape[0], T.prod(r5.shape[1:])))
        #full6 = self.create_unit('imnet_full6',
        #                         ninput=test_shape(r5_vec)[1]))
        #z6 = 0.5 * full6.infer(r5_vec)
        #r6 = relu(z6)

        #full7 = self.create_unit('imnet_full7', ninput=test_shape(r6)[1])
        #z7 = 0.5 * full7.infer(r6)
        #r7 = relu(z7)

        #full8 = self.create_unit('imnet_full8', ninput=test_shape(r7)[1])
        #z8 = full8.infer(r7)

        #output = softmax(z8, axis=1)

        self.imagenet = MachinePart(locals())

    def define_coarse_stack(self, imnet_feats):
        full1 = self.create_unit('full1', ninput=test_shape(imnet_feats)[1])
        f_1 = relu(full1.infer(imnet_feats))
        f_1_drop = random_zero(f_1, 0.5)
        f_1_mean = 0.5 * f_1

        full2 = self.create_unit('full2', ninput=test_shape(f_1_mean)[1])

        f_2_drop = full2.infer(f_1_drop)
        f_2_mean = full2.infer(f_1_mean)

        # prediction
        (h, w) = self.coarse_output_size
        pred_drop = f_2_drop.reshape((self.bsize, h, w))
        pred_mean = f_2_mean.reshape((self.bsize, h, w))

        self.coarse = MachinePart(locals())

    def define_fine_stack(self, x0):
        # crop top part of image out (no depth)
        x0_crop = x0[:,:,56:,:]
        x0_pproc = (x0_crop - self.meta.images_mean) \
                   * self.meta.images_istd

        conv_s2_1 = self.create_unit('conv_s2_1')
        z_s2_1    = relu(conv_s2_1.infer(x0_pproc))

        pool_s2_1 = self.create_unit('pool_s2_1')
        (p_s2_1, s_s2_1) = pool_s2_1.infer(z_s2_1)
        
        # concat input features with coarse prediction
        (h, w) = self.coarse_output_size
        coarse_drop = self.coarse.pred_drop.reshape((self.bsize, 1, h, w))
        coarse_mean = self.coarse.pred_mean.reshape((self.bsize, 1, h, w))
        p_1_concat_drop = T.concatenate(
                              (coarse_drop[:, :, :-1, 1:-1],
                               p_s2_1[:, 1:, :, :]),
                              axis=1)
        p_1_concat_mean = T.concatenate(
                              (coarse_mean[:, :, :-1, 1:-1],
                               p_s2_1[:, 1:, :, :]),
                              axis=1)

        conv_s2_2 = self.create_unit('conv_s2_2')
        z_s2_2_drop = relu(conv_s2_2.infer(p_1_concat_drop))
        z_s2_2_mean = relu(conv_s2_2.infer(p_1_concat_mean))

        conv_s2_3 = self.create_unit('conv_s2_3')
        z_s2_3_drop = conv_s2_3.infer(z_s2_2_drop)
        z_s2_3_mean = conv_s2_3.infer(z_s2_2_mean)

        # prediction
        pred_drop = z_s2_3_drop[:,0,:,:]
        pred_mean = z_s2_3_mean[:,0,:,:]

        self.fine = MachinePart(locals())

    def define_cost(self, pred, y0, m0):
        bsize = self.bsize
        npix = int(np.prod(test_shape(y0)[1:]))
        y0_target = y0.reshape((self.bsize, npix))
        y0_mask = m0.reshape((self.bsize, npix))
        pred = pred.reshape((self.bsize, npix))

        p = pred * y0_mask
        t = y0_target * y0_mask

        d = (p - t)

        nvalid_pix = T.sum(y0_mask, axis=1)
        depth_cost = (T.sum(nvalid_pix * T.sum(d**2, axis=1))
                         - 0.5*T.sum(T.sum(d, axis=1)**2)) \
                     / T.maximum(T.sum(nvalid_pix**2), 1)

        return depth_cost

