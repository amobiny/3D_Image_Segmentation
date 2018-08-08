import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d, deconv_3d, BN_Relu_conv_3d, max_pool, batch_norm, Relu, drop_out
from utils import get_num_channels


class DenseNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=5,
                 num_blocks=(4, 5, 7, 10, 12),  # number of bottleneck blocks at each level
                 bottom_convs=15):              # number of convolutions at the bottom of the network

        super(DenseNet, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_blocks = num_blocks
        self.bottom_convs = bottom_convs
        self.k = self.conf.growth_rate
        self.down_conv_factor = 2
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('DenseNet'):
            feature_list = list()
            shape_list = list()

            with tf.variable_scope('input'):
                x = conv_3d(x, filter_size=3, num_filters=2 * self.k, stride=2, layer_name='input_layer',
                            add_batch_norm=False, is_train=self.is_training)
                print('{}: {}'.format('input_layer', x.get_shape()))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        level = self.dense_block(x, self.num_blocks[l])
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
                        x = self.up_conv(x, self.num_blocks[l], out_shape=out_shape)
                        stack = tf.concat((x, f), axis=-1)
                        print('{}: {}'.format('Decoder_level' + str(l + 1), x.get_shape()))
                        x = self.dense_block(stack, self.num_blocks[l])
                        print('{}: {}'.format('Dense_block_level' + str(l + 1), x.get_shape()))
                        stack = tf.concat((stack, x), axis=-1)

            with tf.variable_scope('output'):
                # x = BN_Relu_conv_3d(x, self.k_size, self.conf.num_cls, 'Output_layer', batch_norm=True,
                #                     is_train=self.is_training)
                # x = tf.nn.dropout(x,self.keep_prob)
                print('{}: {}'.format('out_block_input', stack.get_shape()))
                self.logits = BN_Relu_conv_3d(stack, 1, self.conf.num_cls, 'Output_layer', add_batch_norm=True,
                                              is_train=self.is_training)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_convolutions):
        x = layer_input
        layers = []
        for i in range(num_convolutions):
            layer = BN_Relu_conv_3d(inputs=x,
                                    filter_size=3,
                                    num_filters=self.conf.start_channel_num,
                                    layer_name='conv_' + str(i + 1),
                                    add_batch_norm=self.conf.use_BN,
                                    use_relu=True,
                                    is_train=self.is_training)
            layer = tf.nn.dropout(layer, self.keep_prob)
            layers.append(layer)
            x = tf.concat((x, layer), axis=-1)
        return tf.concat(layers, axis=-1)

    def bottleneck_block(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training, scope='batch1')
            x = Relu(x)
            x = conv_3d(x, filter_size=1, num_filters=4 * self.k, layer_name='conv1')
            x = drop_out(x, keep_prob=self.keep_prob)

            x = batch_norm(x, is_training=self.is_training, scope='batch2')
            x = Relu(x)
            x = conv_3d(x, filter_size=3, num_filters=self.k, layer_name='conv2')
            x = drop_out(x, keep_prob=self.keep_prob)
            return x

    def transition_down(self, x):
        num_out_channels = get_num_channels(x)
        x = BN_Relu_conv_3d(inputs=x,
                            filter_size=1,
                            num_filters=num_out_channels,
                            layer_name='conv_down',
                            stride=1,
                            add_batch_norm=self.conf.use_BN,
                            is_train=self.is_training,
                            use_relu=True)
        x = tf.nn.dropout(x, self.keep_prob)
        x = max_pool(x, self.conf.pool_filter_size, name='maxpool')
        return x

    def transition_up(self, x, num_out_channels, out_shape):
        num_out_channels = num_out_channels * self.conf.start_channel_num  # x.get_shape()[-1]
        x = deconv_3d(inputs=x,
                      filter_size=3,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      add_batch_norm=False,
                      is_train=self.is_training,
                      out_shape=out_shape)
        return x
