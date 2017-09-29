import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import argparse
import json
import glob
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
import random
import collections
import math
import time
import skimage.io as skio
import matplotlib.pyplot as plt

from PIL import Image

import alex_net

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 224

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        contents = tf.Print(contents, [paths])
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = raw_input[:,:width//2,:]
        b_images = raw_input[:,width//2:,:]

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.random_crop(r, [CROP_SIZE, CROP_SIZE, 3])
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def read_input(fnq, size=224, batch_size=16):
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
    
    pos_im = preprocess(tf.image.convert_image_dtype(pos_im, tf.float32))
    neg_im = preprocess(tf.image.convert_image_dtype(neg_im, tf.float32))
    
    return pos_im, neg_im


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def discriminate(inp, num_classes=56, is_training=True):
    with tf.variable_scope("dis"):
        with slim.arg_scope(alex_net.alexnet_bn_arg_scope()):
            outputs = alex_net.alexnet_v2(
                inp, is_training=is_training, spatial_squeeze=True, num_classes=num_classes)
            return outputs
    

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]    
        
def l1_loss(im1, im2):
    return tf.reduce_mean(tf.abs(im1 - im2))

def linf_loss(im1, im2):
    return tf.reduce_max(tf.abs(im1 - im2))

        
def png_save(im, fname):
    # Saves -1, 1 ims to png files
    im = (im + 1) / 2

    im = np.clip(im, 0, 1)
    im = np.uint8(im*255)
    img = Image.fromarray(im)
    img.save(fname, "PNG")
    return
   
def main():
    BATCH_SIZE = 32
    max_labels = 1
    im_size = 224
    LAMBDA = 300
    
    photo_dir = "/home/ahliu/forensics/dres_after_photo/"
    
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")
    
    num_files = 11 # 12 is the validation
    shard_dir = "/home/ahliu/forensics/crop_medifor_shard"
    fname = [os.path.join(shard_dir, "shard%d.train"%(i)) for i in range(num_files)]
    test_fname = [os.path.join(shard_dir, "shard11.train")]
    
    sip = [tf.train.string_input_producer([name]) for name in fname]
    targets = [read_input(_, size=224) for _ in sip]
    g_targets = [read_input(_, size=256) for _ in sip]
    
    test_sip = [tf.train.string_input_producer([name]) for name in test_fname]
    test_targets_256 = [read_input(_, size=256) for _ in test_sip]
    test_targets_224 = [read_input(_, size=224) for _ in test_sip]
    
    test_pos_im, test_neg_im = tf.train.shuffle_batch_join(test_targets_224, batch_size=BATCH_SIZE,
                                             capacity = 50 * BATCH_SIZE, min_after_dequeue=BATCH_SIZE)
    gen_test_pos_im, gen_test_neg_im = tf.train.shuffle_batch_join(test_targets_256, batch_size=BATCH_SIZE,
                                             capacity = 50 * BATCH_SIZE, min_after_dequeue=BATCH_SIZE)
    train_pos, train_neg = tf.train.shuffle_batch_join(targets, batch_size=BATCH_SIZE,
                                             capacity = 50 * BATCH_SIZE, min_after_dequeue=BATCH_SIZE)
    
    # Goal is to get neg images to classify as positive.
    gen_pos, gen_neg = tf.train.shuffle_batch_join(g_targets, batch_size=BATCH_SIZE,
                                             capacity = 50 * BATCH_SIZE, min_after_dequeue=BATCH_SIZE)

    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    pos_logits = discriminate(train_pos, max_labels)
    neg_logits = discriminate(train_neg, max_labels)
    
    pos_accuracy = tf.reduce_mean(tf.cast(tf.greater(pos_logits, 0), dtype=tf.float32))
    neg_accuracy = tf.reduce_mean(tf.cast(tf.less(neg_logits, 0), dtype=tf.float32))
    
    d_loss = 1.2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits,
                                                                    labels=tf.ones_like(pos_logits)))
    d_loss += 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits,
                                                                     labels=tf.zeros_like(neg_logits)))
    
    with tf.variable_scope("gen"):
        gen_res = create_generator(gen_neg, 3)
        gen_d = gen_neg + gen_res
        gen = gen_d[:, 0:im_size, 0:im_size, :]
        
    with tf.variable_scope("gen", reuse=True):
        gen_test_res = create_generator(gen_test_neg_im, 3)
        gen_test_d = gen_test_neg_im + gen_test_res
        gen_2 = gen_test_d[:, 0:im_size, 0:im_size, :]
        
    fake_logits = discriminate(gen, max_labels)
    fake_logits_test = discriminate(gen_2, max_labels)
    
    g_fake_accuracy_test = tf.reduce_mean(tf.cast(tf.greater(fake_logits_test, 0), dtype=tf.float32))
    
    g_fake_accuracy = tf.reduce_mean(tf.cast(tf.greater(fake_logits, 0), dtype=tf.float32))
    g_loss_on_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                         labels=tf.ones_like(fake_logits)))
    
    # fake_logits are the negative gen, the classifier should identify them as such.
    d_loss = d_loss + 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                             labels=tf.zeros_like(fake_logits)))
    g_loss = g_loss_on_d + LAMBDA * l1_loss(gen_d, gen_neg)
    
    pos_test_logits = discriminate(test_pos_im, max_labels)
    neg_test_logits = discriminate(test_neg_im, max_labels)
    
    
    pos_test_accuracy = tf.reduce_mean(tf.cast(tf.greater(pos_test_logits, 0), dtype=tf.float32))
    neg_test_accuracy = tf.reduce_mean(tf.cast(tf.less(neg_test_logits, 0), dtype=tf.float32))
    
    # TRAIN OPTIMIZERS
    t_vars = tf.trainable_variables()
    
    g_vars = [v for v in t_vars if "gen" in v.name]
    g_ts = tf.train.AdamOptimizer(0.00003).minimize(g_loss, var_list=g_vars)
    
    d_vars = [v for v in t_vars if "dis" in v.name]
    d_ts = tf.train.AdamOptimizer(0.00003).minimize(d_loss, var_list=d_vars)
    
    saver = tf.train.Saver()
    sv = tf.train.Supervisor()
    
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    
    with sv.managed_session(config=config) as sess:
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            saver.restore(sess, a.checkpoint)
        
        """
        gobj = sess.run(gen_pos)
        n = gobj[0]
        
        n = (n + 1) / 2
        
        n = np.clip(n, 0, 1)
        n = np.uint8(n*255)
                
        im = Image.fromarray(n)
                
        im.save("b_test.png", "PNG")
        1/0.
        """
            
        """
        # visualize2
        
        before_obj, res_obj, after_obj = sess.run([gen_test_neg_im, gen_test_res, gen_test_d])
        
        before = before_obj[0]
        after = after_obj[0]
        res = res_obj[0]
        
        before = (before + 1) / 2
        after = (after + 1) / 2

        before = np.clip(before, 0, 1)
        after = np.clip(after, 0, 1)
        
        before = np.uint8(before * 255)
        after = np.uint8(after * 255)
             
        
        png_save(before, "before.png")
        png_save(after, "after.png")
        skio.imsave("before.jpg", before)
        skio.imsave("after.jpg", after)
        # skio.imsave("res.jpg", res)
        
        # before_im = skio.imread("before.jpg") / 255.
        # print np.sum(np.abs(before_im-before))
        
        
        1/0.
        """
            
        """
        # visualize
        count = 0
        base_dir = "/data/efros/ahliu/medifor_crop_test"
        after_dir = os.path.join(base_dir, "after")
        before_dir = os.path.join(base_dir, "before")
        res_dir = os.path.join(base_dir, "res")
        
        for count in range(5):
            before_obj, res_obj, after_obj = sess.run([gen_test_neg_im, gen_test_res, gen_test_d])
            for ind in range(BATCH_SIZE):
                before = before_obj[ind]
                res = res_obj[ind]
                after = after_obj[ind]

                a_fname = os.path.join(after_dir, "%d_%d.png"%(count, ind))
                b_fname = os.path.join(before_dir, "%d_%d.png"%(count, ind))
                r_fname = os.path.join(res_dir, "%d_%d.npy"%(count, ind))
                
                img = Image.fromarray(before_im)
                img.save(b_fname, "PNG")

                img = Image.fromarray(after_im)
                img.save(a_fname, "PNG")

                np.save(r_fname, res)
        1/0.
        """

        """
        # Test
        pt = 0.
        nt = 0.
        gt = 0.
        for i in range(300):
            ptl = sess.run(pos_test_accuracy)
            ntl = sess.run(neg_test_accuracy)
            gta = sess.run(g_fake_accuracy_test)
            pt += ptl
            nt += ntl
            gt += gta
        print "pos: " + str(pt / 300)
        print "neg: " + str(nt / 300)
        print "gen test: " + str(gt / 300)
        1/0.
        """
        
        for i in range(100000):
            g_f_a, p_a, n_a, dl_, gl_ = sess.run([g_fake_accuracy, pos_accuracy, neg_accuracy, d_loss, g_loss])
            if i % 100 == 0:
                count = 0
                pos_total = 0.0
                neg_total = 0.0
                for k in range(10):
                    pos_t_acc, neg_t_acc = sess.run([pos_test_accuracy, neg_test_accuracy])
                    pos_total += float(pos_t_acc)
                    neg_total += float(neg_t_acc)
                    count += 1
                print "ave_pos_test accuracy: " + str(pos_total / count)
                print "ave_neg_test accuracy: " + str(neg_total / count)
                g_im, g_d = sess.run([gen_neg, gen_d])
                png_save(g_im[0], "before_.png")
                png_save(g_d[0], "after_.png")
                out = np.abs(g_im - g_d)
                print i, dl_, gl_
                print "d_pos_accuracy: " + str(p_a)
                print "d_neg_accuracy: " + str(n_a)
                print "g_fake_accuracy (% gen image classified positively): " + str(g_f_a)
                print "gen_mean diff: " + str(out.mean())
                print "gen_max diff: " + str(out.max())
                if True:
                    print "SAVING: " + saver.save(sess, "/data/efros/ahliu/checkpoints/cam_model/medifor_corp", global_step=i)
                else:
                    saver.save(sess, "/data/efros/ahliu/checkpoints/temp_checkpoints/ckpt", global_step=i)
            sess.run([g_ts, d_ts])
            for i in range(10):
                sess.run([g_ts])
main()


