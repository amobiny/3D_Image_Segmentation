import tensorflow as tf
from base_model import BaseModel
from ops import conv_3d
from utils import get_num_channels


class VNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=4,
                 num_convs=(1, 2, 3, 3),
                 bottom_convs=3):

        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
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
                with tf.variable_scope('Encoder/level_' + str(l + 1)):
                    x = self.conv_block(self.x, self.num_convs[l])
                    features.append(x)
                    x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.conv_block(x, self.bottom_convs)

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

    def down_conv(self, x):
        num_out_channels = get_num_channels(x)*2
        x = conv_3d(x, 2, num_out_channels, 'conv_down', 2, self.is_training)
        return x

