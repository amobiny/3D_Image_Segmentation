import tensorflow as tf
from base_model import BaseModel
from ops import conv_3d
from utils import get_num_channels


class VNet(BaseModel):
    def __init__(self, sess, conf, num_levels, num_convs):
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.down_conv_factor = 2
        # BaseModel.__init__(self, sess, conf)
        super(VNet, self).__init__(sess, conf)
        # super().__init__(sess, conf)  Python3
        self.build_network()
        self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('V-Net'):
            features = list()

            for l in range(self.num_levels):
                with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                    x = self.conv_block(self.x, self.num_convs[l])
                    features.append(x)
                    with tf.variable_scope('conv_down'):
                        x = self.down_convolution(x, kernel_size=[2, 2, 2])

    def conv_block(self, layer_input, num_convolutions):
        x = layer_input
        n_channels = get_num_channels(x)
        for i in range(num_convolutions):
            x = conv_3d(x, self.k_size, n_channels, 'conv_'+str(i+1), self.is_training, use_relu=False)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, self.keep_prob)
        return x

    def down_convolution(self, x, factor, kernel_size):
        num_channels = get_num_channels(x)
        filter = kernel_size + [num_channels, num_channels * factor]
        x = conv_3d(x, 2, )
        return x