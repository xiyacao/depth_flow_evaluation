import os
import sys
import numpy as np

from PIL import Image

import net

def main():
    # location of depth module, config and parameters
    module_fn = 'models/depth_kitti.py'
    config_fn = 'models/depth_kitti.conf'
    params_dir = 'weights/depth_kitti'

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    rgb = Image.open('demo_kitti_rgb.jpg')
    rgb = rgb.resize((612, 184), Image.BICUBIC)

    # build depth inference function and run
    rgb_imgs = np.asarray(rgb).reshape((1, 184, 612, 3))
    pred_depths = machine.infer_depth(rgb_imgs)

    # save prediction
    (m, M) = (pred_depths.min(), pred_depths.max())
    depth_img_np = (pred_depths[0] - m) / (M - m)
    depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
    depth_img.save('demo_kitti_depth_prediction.png')

    import matplotlib.pyplot as plt
    plt.ion()
    import ipdb
    ipdb.set_trace()
    pass


if __name__ == '__main__':
    main()
