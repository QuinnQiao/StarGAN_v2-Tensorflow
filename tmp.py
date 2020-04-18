import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)