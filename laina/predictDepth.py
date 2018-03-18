import numpy as np
import tensorflow as tf
from PIL import Image

import models

def predictDepth(image_list, pwd):

    depth = []
    # Default input size
    height = 128
    width = 416
    channels = 3
    batch_size = 1
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, pwd + '/laina/NYU_FCRN.ckpt')

        # Use to load from npy file
        #net.load(model_data_path, sess) 
        for image in image_list:
            # Read image
            img = Image.open(image)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            depth.append(pred[0,:,:,0])
    depth = np.array(depth)
    np.save(pwd + "/laina/output/depth.npy", depth)
        
    return depth
        
                

        
#depth = predictDepth('NYU_FCRN.ckpt', ['demo_nyud_rgb.jpg','demo_nyud_rgb.jpg'])


