import cv2
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

# The gate convolution is made with reference to Deepfillv1(https://github.com/JiahuiYu/generative_inpainting)

@tf.keras.utils.register_keras_serializable()
@tf.keras.layers.Layer
def GateConv2D(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True, training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x_in, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.keras.layers.Conv2D(
        cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)(x_in)
    if use_lrn:
        x = tf.nn.local_response_normalization(x, bias=0.00005)
    if activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x)

    g = tf.keras.layers.Conv2D(
        cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name + '_g')(x_in)

    x = tf.multiply(x, g)
    return x, g

@tf.keras.utils.register_keras_serializable()
@tf.keras.layers.Layer
def GateDeconv2D(input_shape, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv", training=True):
    with tf.keras.backend.name_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.keras.backend.random_normal([k_h, k_w, output_shape[-1], input_shape[-1]], stddev=stddev)

        deconv = tf.keras.layers.Conv2DTranspose(output_shape[-1], (k_h, k_w), (d_h, d_w),
                    padding='same', output_padding=None, strides=(d_h, d_w),
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev))(input_shape)

        biases = tf.keras.backend.zeros([output_shape[-1]])
        deconv = tf.reshape(tf.keras.backend.bias_add(deconv, biases), deconv.shape)
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.keras.layers.Conv2DTranspose(output_shape[-1], (k_h, k_w), (d_h, d_w),
                    padding='same', output_padding=None, strides=(d_h, d_w),
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev))(input_shape)
        b = tf.keras.backend.zeros([output_shape[-1]])
        g = tf.reshape(tf.keras.backend.bias_add(g, b), deconv.shape)
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g, deconv)

        return deconv, g
