import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

class CropGenerator(object):
    
    def get_random_crop_index(self):
        x_crop = tf.to_int32(
            tf.random_uniform([self.batch_size]) * self.x_num_choice_crops)
        y_crop = tf.to_int32(
            tf.random_uniform([self.batch_size]) * self.y_num_choice_crops)
        return x_crop, y_crop
    
    def __init__(self, size_of_random, im_size, crop_size=64, batch_size=10):
        # Generates a float that indicates a good cropping
        # size_of_random: How many bits to use
        # img_inp: Input tensor to find crops for
        # size: Size of crops to produce
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.input_size = size_of_random
        self.build_net(batch_size, im_size, crop_size)
    
    def build_net(self, batch_size, im_size, crop_size):
        self.x_num_choice_crops = int(im_size[1] - crop_size)
        self.y_num_choice_crops = int(im_size[0] - crop_size)
        with tf.variable_scope("G_net"):
            net = slim.fully_connected(
                tf.random_uniform([batch_size, self.input_size]), 200)
            self.x_weights = slim.fully_connected(net, self.x_num_choice_crops,
                                                  activation_fn=tf.sigmoid)
            self.y_weights = slim.fully_connected(net, self.y_num_choice_crops,
                                                  activation_fn = tf.sigmoid)
    
    def get_REIN_training_crop(self, e=.2):
        # e represents the explore factor
        con = tf.constant(e)
        rand = tf.random_uniform([1])[0]
        return tf.cond(
            tf.greater(con, rand),
            lambda: self.get_random_crop_index(),
            lambda: self.choose_crop_index())
    
    def choose_crop_index(self):
        y_crop = tf.argmax(self.y_weights, 1)
        x_crop = tf.argmax(self.x_weights, 1)
        return tf.to_int32(x_crop), tf.to_int32(y_crop)
    
    def get_crops(self, im_inp, x, y):
        # im_inp is a tensor that matches (batch_size, im_size)
        # x, y represent a choose_crop_index call or random_crop call
        return tf.stack([tf.image.crop_to_bounding_box(
                  im_inp[i], y[i], x[i], self.crop_size, self.crop_size)
                  for i in range(self.batch_size)])
    
    def loss_fn(self, D_fake):
        # D_fake repesent the discriminator's prediction of generate() 
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake, labels=tf.ones_like(D_fake))
        
    def train(self, action_x, action_y, reward):
        action_x = tf.to_int32(action_x)
        action_y = tf.to_int32(action_y)
        responsible_x = tf.stack([tf.slice(self.x_weights[i], [action_x[i]], [1]) for
                                  i in range(self.batch_size)])
        
        responsible_y = tf.stack([tf.slice(self.y_weights[i], [action_y[i]], [1]) for
                                  i in range(self.batch_size)])
        
        g_loss = -(tf.log(responsible_x * responsible_y)*reward)
        g_optim = tf.train.AdamOptimizer(0.003) \
            .minimize(g_loss)
        # t_vars = tf.trainable_variables()
        # g_vars = [v for v in t_vars if "G_net" in v.name]
        # g_loss = self.loss_fn(D_fake)
        # g_optim = tf.train.AdamOptimizer(0.003) \
        #      .minimize(g_loss, var_list=g_vars)
        return g_optim
            
        

def dummy_discrim(x_crop, y_crop, correct_x=0, correct_y=0):
    x_crop = tf.mod(x_crop, 8)
    y_crop = tf.mod(y_crop, 8)
    return tf.where(tf.logical_and(tf.equal(x_crop, correct_x),
                                   tf.equal(y_crop, correct_y)),
                    tf.ones_like(x_crop, dtype=tf.float32),
                    -1 * tf.ones_like(x_crop, dtype=tf.float32))

"""
gen = CropGenerator(300, (256, 256), 64)
im = tf.placeholder(tf.float32, [10, 256, 256, 3])

x, y = gen.get_REIN_training_crop(1.)
d_fake = dummy_discrim(x, y)
train = gen.train(x, y, d_fake)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    print sess.run([x, y, train])
    
gen = CropGenerator(300, (256, 256), 64)

x, y = gen.choose_crop_index()
d_fake = dummy_discrim(x, y)
train = gen.train(x, y, d_fake)

random_x, random_y = gen.get_random_crop_index()
d_fake_random = dummy_discrim(random_x, random_y)
train_random = gen.train(random_x, random_y, d_fake)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
e = 0
for i in range(1000):
    if random.random() < e:
        print "RANDOM"
        print sess.run([random_x, random_y, train_random])
    else:
        print sess.run([x, y, train])

count = 0   
for i in range(500):
    [x_val, y_val, _] = sess.run([x, y, train])
    for i in range(10):
        if x_val[i] % 8 == 0 and y_val[i] % 8 == 0:
            count += 1
"""