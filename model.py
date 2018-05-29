import os


class Unet_3D(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3, 3)
        self.pool_size = (2, 2, 2)
        self.axis, self.channel_axis = (1, 2, 3), 4
        self.input_shape = [self.conf.batch, self.conf.depth, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [self.conf.batch, self.conf.depth, self.conf.height, self.conf.width]
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.savedir):
            os.makedirs(conf.savedir)