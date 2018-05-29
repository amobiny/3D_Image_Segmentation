import os
import tensorflow as tf


class Unet_3D(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.axis, self.channel_axis = (1, 2, 3), 4
        self.input_shape = [self.conf.batch, self.conf.depth, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [self.conf.batch, self.conf.depth, self.conf.height, self.conf.width]
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.savedir):
            os.makedirs(conf.savedir)
        self.create_placeholders()
        self.build_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.keep_prob = tf.placeholder(tf.float32)

    def inference(self):
        self.predictions = self.inference(self.x)
