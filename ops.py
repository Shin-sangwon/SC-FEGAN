import cv2
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

class GateConv2D(tf.keras.layers.Layer):
    def __init__(self, cnum, ksize, stride=1, rate=1, name='conv', padding='SAME', activation='leaky_relu', use_lrn=True, training=True):
        super(GateConv2D, self).__init__(name=name)
        assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        if padding == 'SYMMETRIC' or padding == 'REFELECT':
            p = int(rate * (ksize - 1) / 2)
            self.padding = padding
            self.p = p
            padding = 'VALID'
        else:
            self.padding = padding
            self.p = 0
        self.conv = tf.keras.layers.Conv2D(
            cnum, ksize, stride, dilation_rate=rate,
            activation=None, padding=padding, name=name)
        self.use_lrn = use_lrn
        self.activation = activation
        self.lrn = tf.keras.layers.LocallyConnected2D(1, 1)  # For local response normalization

    def call(self, inputs):
        x_in = inputs
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            x = tf.pad(x_in, [[0, 0], [self.p, self.p], [self.p, self.p], [0, 0]], mode=self.padding)
            padding = 'VALID'
        else:
            x = x_in
            padding = self.padding
        x = self.conv(x)
        if self.use_lrn:
            x = self.lrn(x)
        if self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x)

        g = tf.keras.layers.Conv2D(
            cnum, ksize, stride, dilation_rate=rate,
            activation=tf.nn.sigmoid, padding=padding, name=name + '_g')(x_in)

        x = tf.multiply(x, g)
        return x, g


class GateDeconv2D(tf.keras.layers.Layer):
    def __init__(self, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv", training=True):
        super(GateDeconv2D, self).__init__(name=name)
        self.k_h = k_h
        self.k_w = k_w
        self.d_h = d_h
        self.d_w = d_w
        self.stddev = stddev
        self.deconv = tf.keras.layers.Conv2DTranspose(
            output_shape[-1], (k_h, k_w), (d_h, d_w),
            padding='same', output_padding=None, strides=(d_h, d_w),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev))
        self.biases = tf.Variable(tf.zeros([output_shape[-1]]))
        self.g = tf.keras.layers.Conv2DTranspose(
            output_shape[-1], (k_h, k_w), (d_h, d_w),
            padding='same', output_padding=None, strides=(d_h, d_w),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev))
        self.b = tf.Variable(tf.zeros([output_shape[-1]]))

    def call(self, inputs):
        input_shape, output_shape = inputs
        with tf.keras.backend.name_scope(self.name):
            w = tf.keras.backend.random_normal([self.k_h, self.k_w, output_shape[-1], input_shape[-1]], stddev=self.stddev)
            deconv = self.deconv(input_shape)
            deconv = tf.reshape(tf.keras.backend.bias_add(deconv, self.biases), deconv.shape)
            deconv = tf.nn.leaky_relu(deconv)

            g = self.g(input_shape)
            g = tf.reshape(tf.keras.backend.bias_add(g, self.b), deconv.shape)
            g = tf.nn.sigmoid(deconv)

            deconv = tf.multiply(g, deconv)

            return deconv, g
