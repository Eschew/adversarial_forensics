import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

class CropDiscriminator(object):
    def __init__(self, real_placeholder, fake_placeholder):
        self.real_placeholder = real_placeholder
        self.fake_placeholder = fake_placeholder
        self.build_net()
        
    def build_net(self):
        self.real_prediction = self.discriminate(self.real_placeholder)
        self.fake_prediction = self.discriminate(self.fake_placeholder)
        
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_prediction,
                                                    labels=tf.ones_like(self.real_prediction))) 
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_prediction,
                                                    labels=tf.zeros_like(self.fake_prediction)))
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
    def train(self):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D_net' in var.name]
        return tf.train.AdamOptimizer(0.0003).minimize(
            self.d_loss, var_list=self.d_vars)
    
    def get_g_logits(self):
        return self.fake_prediction
    
    def discriminate(self, inp):
        # Train
        with tf.variable_scope("D_net"):
            net = slim.conv2d(inp, 32, 5)
            net = slim.conv2d(net, 32, 5)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 16, 5)
            net = slim.conv2d(net, 16, 5)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 8, 5)
            net = slim.conv2d(net, 8, 5)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 4, 5)
            net = slim.conv2d(net, 4, 5)
            net = slim.max_pool2d(net, 2)

            net = slim.flatten(net)

            net = slim.fully_connected(net, 2048)
            net = slim.fully_connected(net, 1, activation_fn=tf.sigmoid)
        return net
"""
real = tf.placeholder(tf.float32, [None, 64, 64, 3])
fake = tf.placeholder(tf.float32, [None, 64, 64, 3])
cd = CropDiscriminator(real, fake)"""
