import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import tensorflow.contrib.slim as slim
import skimage
import skimage.io as skio
import skimage.transform
import scipy
import scipy.misc
import matplotlib.pyplot as plt

import crop_generator
import crop_discriminator

tf.app.flags.DEFINE_string('load_model', "checkpoint",
                           """Load a previous model.""")
tf.app.flags.DEFINE_integer("num_steps", 10000,
                           """Number of steps forward to train model.""")
tf.app.flags.DEFINE_integer('global_step', 0,
                            """Number of steps in the program's lifetime""")

FLAGS = tf.app.flags.FLAGS

im_dir = "images/"
image_files = [f for f in os.listdir(im_dir) if '.jpg' in f]

KERNEL_FILTER_SIZE = 64

def get_compression(im, factor=10):
    buff = io.BytesIO()
    img = Image.fromarray(im)
    img.save(buff, "JPEG", quality=factor)
    return skio.imread(io.BytesIO(buff.getvalue()))

# k x 1920 x 1080 x 3
IMAGES = np.array([skio.imread(os.path.join(im_dir, image_file)) for image_file in image_files] * 5)

def random_crop(im, size):
    x = np.random.randint(0, im.shape[1] - size[1])
    y = np.random.randint(0, im.shape[0] - size[0])
    return im[y:y+size[0], x:x+size[1], :]
    

def random_uncompressed_crop(im, size, scaling=.4, random_transform=False):
    """Random uncompress crop and scale
    """
    scale = np.random.uniform(-scaling, scaling)
    resized_im = scipy.misc.imresize(im, 1. + scale)
    resize_im = random_crop(resized_im, size)
    if random_transform:
        resize_im = skimage.transform.rotate(resize_im, np.random.randint(4)*90)
    return resize_im

def format_training_im(im):
    """Formats the im for input into a model pipeline.
    Encodes as values between 0, 1 which can be non linearized by a relu
    """
    return im / 255.

def inp_im_g(batch):
    curr = None
    indices = np.random.choice(range(len(IMAGES)), batch)
    print indices
    for index in indices:
        resized_im = random_uncompressed_crop(IMAGES[index], size=(512, 512))
        compressed_im = get_compression(resized_im)
        if type(curr) != type(None):
            curr = np.stack(curr + [resized_im])
        else:
            curr = np.array([resized_im])
        print curr.shape
    return curr

def unformat_training_im(im):
    im = (im / np.max(im))*255.
    return im.astype(np.uint8)

def zero_crop(im, size):
    return im[:size[1], :size[0]]

def get_positive_batch(batch_size):
    curr = None
    for i in range(batch_size):
        index = np.random.randint(len(image_files))
        resized_im = random_uncompressed_crop(IMAGES[index], size=(512, 512))
        compressed_im = get_compression(resized_im)
        new_ims = np.stack([zero_crop(compressed_im,
                                      [KERNEL_FILTER_SIZE, KERNEL_FILTER_SIZE])])
        if type(curr) != type(None):
            curr = np.vstack([curr] + [new_ims])
        else:
            curr = new_ims
    return curr, np.ones((curr.shape[0], 1), dtype=np.float)

def batchify(im, size):
    k = []
    for y in range(im.shape[0]-size[1]):
        for x in range(im.shape[1]-size[0]):
            im_portion = im[y:y+size[1], x:x+size[0], :]
            k.append(im_portion)
    return k

def main(argv=None):
    # Fed into g-network
    fake_im_inp = tf.placeholder(tf.float32, [10, 512, 512, 3])
    # Fed into d-network as real
    real_inp = tf.placeholder(tf.float32, [None, KERNEL_FILTER_SIZE, KERNEL_FILTER_SIZE, 3])
    
    
    gen = crop_generator.CropGenerator(10, (512, 512), crop_size=KERNEL_FILTER_SIZE, batch_size=10)
    
    x, y = gen.get_REIN_training_crop()
    crops = gen.get_crops(fake_im_inp, x, y)
    
    dis = crop_discriminator.CropDiscriminator(real_inp, crops)
    g_reward = dis.get_g_logits()
    g_train = gen.train(x, y, g_reward)
    d_train = dis.train()
    
    d_loss = dis.d_loss
    
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_name = FLAGS.load_model+"-"+str(FLAGS.global_step)
    if tf.train.checkpoint_exists(model_name):
        print "RESTORED MODEL"
        saver.restore(sess, model_name)
        
    for i in range(FLAGS.global_step, FLAGS.global_step + FLAGS.num_steps):
        fake_im = format_training_im(inp_im_g(10)) 
        real = format_training_im(get_positive_batch(10)[0])
        
        sess.run(d_train, feed_dict={
            fake_im_inp:fake_im,
            real_inp:real})
        f, r, l = sess.run([dis.fake_prediction, dis.real_prediction, d_loss], feed_dict={
            fake_im_inp:fake_im,
            real_inp:real})
        print np.mean(f), np.mean(r), l
        """
        if i%2 == 0:
            l = sess.run(d_loss, feed_dict={
                fake_im_inp:fake_im,
                real_inp:real})
            f, r = sess.run([dis.fake_prediction, dis.real_prediction], feed_dict={
                fake_im_inp:fake_im,
                real_inp:real})
            print f, r
            print "SAVING: " + saver.save(sess, FLAGS.load_model, global_step=i)
            print(i, l)
        
        if i%3 == 0:
            sess.run(d_train, feed_dict={
                    fake_im_inp:fake_im,
                    real_inp:real})
        sess.run(g_train, feed_dict={
                    fake_im_inp:fake_im,
                    real_inp:real})
                    """
        
        
if __name__ == '__main__':
    tf.app.run()
    
