import os
import sys
import numpy as np
from PIL import Image
import net

def predictDepth(filelist, pwd):
	image_w = 416
	image_h = 218
    # location of depth module, config and parameters
    module_fn = pwd + '/eigen/models/depth_kitti.py'
    config_fn = pwd + '/eigen/models/depth_kitti.conf'
    params_dir = pwd + '/eigen/weights/depth_kitti'

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)
    rgb_imgs = []
    # demo image
    for filename in filelist:
    	rgb = Image.open(filename)
    	rgb = rgb.resize((image_w, image_h), Image.BICUBIC)
		rgb_imgs.append(rgb)
    	# build depth inference function and run
    rgb_imgs = np.asarray(rgb_imgs).reshape((len(filelist), image_h, image_w, 3))
    pred_depths = machine.infer_depth(rgb_imgs)

    # save prediction
    # (m, M) = (pred_depths.min(), pred_depths.max())
    # depth_img_np = (pred_depths[0] - m) / (M - m)
    np.save(pwd + "/eigen/output/depth.npy", pred_depths)
    return pred_depths