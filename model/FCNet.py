import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d


class FCN(BaseModel):
    def __init__(self, sess, conf):
        # BaseModel.__init__(self, sess, conf)
        super(FCN, self).__init__(sess, conf)
        self.act_fcn = tf.nn.relu
        self.k_size = self.conf.filter_size
        self.pool_size = self.conf.pool_filter_size
        # super().__init__(sess, conf)  Python3
        self.build_network()
        self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('FCN'):
            conv1 = conv_3d(self.x, self.k_size, 16, 'CONV1', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv2 = conv_3d(conv1, self.k_size, 32, 'CONV2', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv3 = conv_3d(conv2, self.k_size, 64, 'CONV3', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv4 = conv_3d(conv3, self.k_size, 128, 'CONV4', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv5 = conv_3d(conv4, self.k_size, 64, 'CONV5', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv6 = conv_3d(conv5, self.k_size, 32, 'CONV6', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv7 = conv_3d(conv6, self.k_size, 16, 'CONV7', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            self.logits = conv_3d(conv7, 1, self.conf.num_cls, 'CONV8', batch_norm=self.conf.use_BN,
                                  is_train=self.is_training)
