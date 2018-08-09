import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d, deconv_3d, BN_Relu_conv_3d, max_pool
from utils import get_num_channels


class Tiramisu(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=5,
                 num_convs=(4, 5, 7, 10, 12),
                 bottom_convs=15):

        super(Tiramisu, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('Tiramisu'):
            feature_list = list()
            shape_list = list()

            with tf.variable_scope('input'):
                x = conv_3d(x, self.k_size, 64, 'input_layer', add_batch_norm=False,
                            add_reg=self.conf.use_reg, is_train=self.is_training)
                # x = tf.nn.dropout(x, self.keep_prob)
                print('{}: {}'.format('input_layer', x.get_shape()))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        level = self.dense_block(x, self.num_convs[l])
                        shape_list.append(tf.shape(level))
                        x = tf.concat((x, level), axis=-1)
                        print('{}: {}'.format('Encoder_level' + str(l + 1), x.get_shape()))
                        feature_list.append(x)
                        x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.dense_block(x, self.bottom_convs)
                print('{}: {}'.format('bottom_level', x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        out_shape = shape_list[l]
                        x = self.up_conv(x, self.num_convs[l], out_shape=out_shape)
                        stack = tf.concat((x, f), axis=-1)
                        print('{}: {}'.format('Decoder_level' + str(l + 1), x.get_shape()))
                        x = self.dense_block(stack, self.num_convs[l])
                        print('{}: {}'.format('Dense_block_level' + str(l + 1), x.get_shape()))
                        stack = tf.concat((stack, x), axis=-1)

            with tf.variable_scope('output'):
                # x = BN_Relu_conv_3d(x, self.k_size, self.conf.num_cls, 'Output_layer', batch_norm=True,
                #                     add_reg=self.conf.use_reg, is_train=self.is_training)
                # x = tf.nn.dropout(x,self.keep_prob)
                print('{}: {}'.format('out_block_input', stack.get_shape()))
                self.logits = BN_Relu_conv_3d(stack, 1, self.conf.num_cls, 'Output_layer', add_batch_norm=True,
                                              add_reg=self.conf.use_reg, is_train=self.is_training)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_convolutions):
        x = layer_input
        layers = []
        # n_channels = get_num_channels(x)
        # if n_channels == self.conf.channel:
        #    n_channels = self.conf.start_channel_num
        for i in range(num_convolutions):
            layer = BN_Relu_conv_3d(inputs=x,
                                    filter_size=self.k_size,
                                    num_filters=self.conf.start_channel_num,
                                    layer_name='conv_' + str(i + 1),
                                    add_batch_norm=self.conf.use_BN,
                                    add_reg=self.conf.use_reg,
                                    use_relu=True,
                                    is_train=self.is_training)
            layer = tf.nn.dropout(layer, self.keep_prob)
            layers.append(layer)
            x = tf.concat((x, layer), axis=-1)
        return tf.concat(layers, axis=-1)

    def down_conv(self, x):
        num_out_channels = get_num_channels(x)
        x = BN_Relu_conv_3d(inputs=x,
                            filter_size=1,
                            num_filters=num_out_channels,
                            layer_name='conv_down',
                            stride=1,
                            add_reg=self.conf.use_reg,
                            add_batch_norm=self.conf.use_BN,
                            is_train=self.is_training,
                            use_relu=True)
        x = tf.nn.dropout(x, self.keep_prob)
        x = max_pool(x, self.conf.pool_filter_size, name='maxpool')
        return x

    def up_conv(self, x, num_out_channels, out_shape):
        num_out_channels = num_out_channels * self.conf.start_channel_num  # x.get_shape()[-1]
        x = deconv_3d(inputs=x,
                      filter_size=3,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      add_reg=self.conf.use_reg,
                      add_batch_norm=False,
                      is_train=self.is_training,
                      out_shape=out_shape)
        return x
