import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print get_available_gpus()
import numpy as np
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
import skimage.io as skio
import matplotlib.pyplot as plt
import pix2pixgen
import pprint


from PIL import Image

import alex_net
import collections

tf.app.flags.DEFINE_string("load_model", "checkpoint",
                           """Load a previous model.""")
tf.app.flags.DEFINE_string("save_model", "",
                           """Destination to save the model to. If blank, load_model-global_step save.""")
tf.app.flags.DEFINE_integer("num_steps", 10000,
                           """Number of steps forward to train model.""")
tf.app.flags.DEFINE_integer('global_step', 0,
                            """Number of steps in the program's lifetime""")
tf.app.flags.DEFINE_boolean('save', True,
                            """Whether to save models.""")

FLAGS = tf.app.flags.FLAGS

def convergeTo0(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.zeros_like(logits)))

def convergeTo1(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.ones_like(logits)))


def linf_loss(im1, im2):
    return tf.reduce_max(tf.abs(im1 - im2))

def l1_loss(im1, im2):
    return tf.reduce_mean(tf.abs(im1-im2))
    
def savePNG(im, fname):
    # Saves -1, 1 ims to png files
    im = (im + 1) / 2

    im = np.clip(im, 0, 1)
    im = np.uint8(im*255)
    img = Image.fromarray(im)
    img.save(fname, "PNG")
    return

        
def raise_arg_error(s):
    raise Exception("Missing/malformed argument '" + s + "'.")

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

class CropGAN(object):
    def __init__(self, params):
        """
        im_size: Size of discriminator input images. (default 224) [<= 256]
        lambda_l1: Determines the weight of the generator's l1-loss (default 30)
        mu_adversarial: Determines the adversarial loss weighing (default 1)
        
        batch_size: Size of batches to feed gen and discrim
        """
        
        self.pp = pprint.PrettyPrinter(indent=2)
        
        # Factors related to reading images
        self.train_shard_files = params.get("train_shard_files", None)
        if not self.train_shard_files:
            raise_arg_error("train_shard_files")
           
        self.test_shard_files = params.get("test_shard_files", None)
        if not self.test_shard_files:
            raise_arg_error("test_shard_files")
        
        self.im_size = params.get("im_size", 224)
        self.batch_size = params.get("batch_size", 32)
        
        self.setup_read()
        
        with tf.variable_scope("crop_gan"):
            self.build_net()
        
        self.save_path = params.get("save_path", None)
        self.checkpoint = params.get("checkpoint", None)
        self.global_step = params.get("global_step", 0)
        self.var = [v for v in tf.trainable_variables() if "crop_gan" in v.name]
        self.saver = tf.train.Saver(var_list=self.var)
        
        self.generator_loss_fn = params.get("gen_loss_fn", l1_loss)
        
    
        
    def read_input(self, fnq, size=224, batch_size=16):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(fnq)
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'width': tf.FixedLenFeature([], tf.int64),
                   'height': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(serialized_example, features=feature)

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        img = tf.image.decode_png(features['image'], channels=3, dtype=tf.uint8)
        img = tf.reshape(img, (width, height, 3))

        pos_im = img[:size, :size, :]
        pos_im.set_shape((size, size, 3))
        neg_im = tf.random_crop(img, (size, size, 3))

        return pos_im, neg_im
        
        
    def save_pos_neg_im(self, sess, directory):
        pos, neg = sess.run([self.train_pos, self.train_neg])
        savePNG(pos[0], os.path.join(directory, "real_pos%s.png"%str(0)))
        savePNG(neg[0], os.path.join(directory, "real_neg%s.png"%str(0)))
        
        
    def build_net(self):
        # Discriminator training
        self.real_pos_crop_logits, self.real_pos_detection_logits = self.discriminator(self.train_pos)
        self.real_neg_crop_logits, self.real_neg_detection_logits = self.discriminator(self.train_neg)
    
        
        """
        self.test_pos_im, self.test_neg_im = tf.train.shuffle_batch_join(test_targets_224, batch_size=self.batch_size,
                                                 capacity = 50 * self.batch_size, min_after_dequeue=self.batch_size)
        self.gen_test_pos_im, self.gen_test_neg_im = tf.train.shuffle_batch_join(test_targets_256, batch_size=self.batch_size,
                                                 capacity = 50 * self.batch_size, min_after_dequeue=self.batch_size)
                                                 """
        
        # Percentage of discriminator positive crops classified positively
        self.d_pos_crop_accuracy = tf.reduce_mean(tf.cast(tf.greater(self.real_pos_crop_logits, 0), dtype=tf.float32))
        # Percentage of discriminator negative crops classified negatively
        self.d_neg_crop_accuracy = tf.reduce_mean(tf.cast(tf.less(self.real_neg_crop_logits, 0), dtype=tf.float32))
        
        # Trainable variables
        self.d_vars = [v for v in tf.trainable_variables() if "dis" in v.name]
        
        
        # Equal number of 0, 1 for training
        self.d_crop_loss_pos = convergeTo1(self.real_pos_crop_logits)
        
        # Converge to 0
        self.d_crop_loss_neg = convergeTo0(self.real_neg_crop_logits)
        
        self.d_crop_loss = self.d_crop_loss_pos + self.d_crop_loss_neg
        
        # Detection Loss + cropping loss accuracy
        self.d_loss = self.d_crop_loss

        self.d_train_step = tf.train.AdamOptimizer(0.0003).minimize(self.d_loss, var_list=self.d_vars)
        
        
        # TEST STRUCTURE
        self.test_real_pos_crop_logits, self.test_real_pos_detection_logits = self.discriminator(self.test_pos_im, is_training=False)
        self.test_real_neg_crop_logits, self.test_real_neg_detection_logits = self.discriminator(self.test_neg_im, is_training=False)
        
        # Percentage of discriminator positive crops classified positively
        self.test_d_pos_crop_accuracy = tf.reduce_mean(tf.cast(tf.greater(self.test_real_pos_crop_logits, 0), dtype=tf.float32))
        # Percentage of discriminator negative crops classified negatively
        self.test_d_neg_crop_accuracy = tf.reduce_mean(tf.cast(tf.less(self.test_real_neg_crop_logits, 0), dtype=tf.float32))
        
        
    def summarize(self, d):
        self.pp.pprint(d)
        
    def summarize_train(self, sess):
        "Prints a train statistics without stepping"
        tracker = collections.defaultdict(float)
        d_l = sess.run([self.d_loss])
        tracker["Discriminator Loss"] = d_l
        
        d_p_c_a, d_n_c_a = sess.run([self.d_pos_crop_accuracy, self.d_neg_crop_accuracy])
        tracker["Discriminator Real Positive Crop Accuracy"] = d_p_c_a
        tracker["Discriminator Real Negative Crop Accuracy"] = d_n_c_a
        
        tracker = dict(tracker)
        
        print "TRAIN STATISTICS (%d):" % self.global_step
        print "================="
        self.summarize(tracker)
        
        
    def summarize_test(self, sess, repeat=20):
        "Prints test statistics"
        tracker = collections.defaultdict(float)
        for i in range(repeat):

            d_p_c_a, d_n_c_a = sess.run([self.test_d_pos_crop_accuracy, self.test_d_neg_crop_accuracy])
            tracker["Discriminator Real Positive Crop Accuracy"] += d_p_c_a
            tracker["Discriminator Real Negative Crop Accuracy"] += d_n_c_a
            
            
        tracker = {k: v / repeat for k, v in tracker.iteritems()}
        
        print "TEST STATISTICS (%d):" % self.global_step
        print "================="
        self.summarize(tracker)
        
    def train_d(self):
        sess.run([self.d_train_step])
        
    def increment_step(self):
        self.global_step += 1
        
    def setup_read(self):
        pp = lambda im: preprocess(tf.image.convert_image_dtype(im, tf.float32))
        
        train_sip = [tf.train.string_input_producer([name]) for name in self.train_shard_files]
        test_sip = [tf.train.string_input_producer([name]) for name in self.test_shard_files]
        
        
        
        test_targets_224 = [self.read_input(_, size=self.im_size) for _ in test_sip]
        targets_224 = [self.read_input(_, size=self.im_size, batch_size=self.batch_size) for _ in train_sip]
        
        self.test_pos_im, self.test_neg_im = tf.train.shuffle_batch_join(test_targets_224, batch_size=self.batch_size,
                                                 capacity = 50 * self.batch_size, min_after_dequeue=self.batch_size)
        
        # Trains discriminator
        self.train_pos, self.train_neg = tf.train.shuffle_batch_join(targets_224, batch_size=self.batch_size,
                                                 capacity = 50 * self.batch_size, min_after_dequeue=self.batch_size)
        
        self.test_pos_im = pp(self.test_pos_im)
        self.test_neg_im = pp(self.test_neg_im)
        
        self.train_pos = pp(self.train_pos)
        self.train_neg = pp(self.train_neg)
        
        
    def discriminator(self, inp, num_classes=2, is_training=True):
        with tf.variable_scope("crop_discriminator"):
            with slim.arg_scope(alex_net.alexnet_bn_arg_scope()):
                crop_logits = alex_net.alexnet_v2(
                    inp, is_training=is_training, spatial_squeeze=True, num_classes=num_classes)
            
        with tf.variable_scope("fake_discriminator"):
            with slim.arg_scope(alex_net.alexnet_bn_arg_scope()):
                detection_logits = alex_net.alexnet_v2(
                    inp, is_training=is_training, spatial_squeeze=True, num_classes=1)
        
        return crop_logits, detection_logits
    
    def save_checkpoint(self, sess, override_save=None, pr=True):
        if not override_save:
            path = self.saver.save(sess, self.save_path, global_step=self.global_step)
        else:
            path = self.saver.save(sess, override_save, global_step=self.global_step)
        if pr:
            print "SAVING: %s" % path
        return path
    
    def load_checkpoint(self, sess, pr=True):
        if not self.checkpoint:
            raise raise_arg_error("checkpoint")
        else:
            ckpt = self.checkpoint + "-" + str(self.global_step)
            self.saver.restore(sess, ckpt)
            print "LOAD MODEL: %s" % ckpt
        
        
        
num_files = 11 # 12 is the validation
shard_dir = "/home/ahliu/forensics/crop_medifor_shard"
im_dump_dir = "/home/ahliu/forensics/l1_loss_im_discrim/"
fname = [os.path.join(shard_dir, "shard%d.train"%(i)) for i in range(num_files)]
test_fname = [os.path.join(shard_dir, "shard11.train")]

params = {}
params['train_shard_files'] = fname
params['test_shard_files'] = test_fname
params['save_path'] = "/data/efros/ahliu/checkpoints/cam_model/discriminator_only_test"
params['checkpoint'] = '/data/efros/ahliu/checkpoints/cam_model/discriminator_only_test'
params['global_step'] = 0

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(params)
cg = CropGAN(params)

saver = tf.train.Saver()
sv = tf.train.Supervisor()
    
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
    
with sv.managed_session(config=config) as sess:
    # cg.load_checkpoint(sess)
    for i in range(400000):
        if i%100 == 0:
            cg.save_checkpoint(sess)
            cg.summarize_train(sess)
            print ""
            cg.summarize_test(sess)
            print ""
            
        cg.train_d()
        cg.increment_step()
        
    