import tensorflow as tf
from base_model import BaseModel
from ops import conv_3d


class FCN_3D(BaseModel):
    def __init__(self, sess, conf):
        # BaseModel.__init__(self, sess, conf)
        super(FCN_3D, self).__init__(sess, conf)
        # super().__init__(sess, conf)  Python3
        self.build_network()
        self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('3D_UNET'):
            conv1 = conv_3d(self.x, self.k_size, 16, 'CONV1', is_train=self.is_training)
            conv2 = conv_3d(conv1, self.k_size, 32, 'CONV2', is_train=self.is_training)
            conv3 = conv_3d(conv2, self.k_size, 64, 'CONV3', is_train=self.is_training)
            conv4 = conv_3d(conv3, self.k_size, 128, 'CONV4', is_train=self.is_training)
            conv5 = conv_3d(conv4, self.k_size, 64, 'CONV5', is_train=self.is_training)
            conv6 = conv_3d(conv5, self.k_size, 32, 'CONV6', is_train=self.is_training)
            conv7 = conv_3d(conv6, self.k_size, 16, 'CONV7', is_train=self.is_training)
            self.logits = conv_3d(conv7, 1, self.conf.num_cls, 'CONV8', is_train=self.is_training, use_relu=False)
