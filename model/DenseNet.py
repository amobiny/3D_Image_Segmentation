import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d, deconv_3d, BN_Relu_conv_3d, max_pool, batch_norm, Relu, drop_out, avg_pool, concatenation
from utils import get_num_channels


class DenseNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=3,
                 num_blocks=(6, 8, 10),  # number of bottleneck blocks at each level
                 bottom_convs=8):  # number of convolutions at the bottom of the network
        assert num_levels == len(num_blocks), "number of levels doesn't match with number of blocks!"
        super(DenseNet, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_blocks = num_blocks
        self.bottom_convs = bottom_convs
        self.k = self.conf.growth_rate
        self.down_conv_factor = 2
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x_input):
        # Building network...
        with tf.variable_scope('DenseNet'):
            feature_list = list()
            shape_list = list()
            x = conv_3d(x_input, filter_size=3, num_filters=2 * self.k, stride=2, layer_name='conv1',
                        add_batch_norm=False, is_train=self.is_training, add_reg=self.conf.use_reg)
            print('conv1 shape: {}'.format(x.get_shape()))
            shape_list.append(tf.shape(x[:, :, :, :, :self.k]))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        x = self.dense_block(x, self.num_blocks[l], scope='DB_' + str(l + 1))
                        feature_list.append(x)
                        print('DB_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        x = self.transition_down(x, scope='TD_' + str(l + 1))
                        print('TD_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        if l != self.num_levels - 1:
                            shape_list.append(tf.shape(x))

            with tf.variable_scope('Bottom_level'):
                x = self.dense_block(x, self.bottom_convs, scope='BottomBlock')
                print('bottom_level shape: {}'.format(x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        out_shape = shape_list[l]
                        x = self.transition_up(x, out_shape=out_shape, scope='TU_' + str(l + 1))
                        print('TU_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        stack = tf.concat((x, f), axis=-1)
                        print('After concat shape: {}'.format(stack.get_shape()))
                        x = self.dense_block(stack, self.num_blocks[l], scope='DB_' + str(l + 1))
                        print('DB_{} shape: {}'.format(str(l + 1), x.get_shape()))

            with tf.variable_scope('output'):
                out_filters = x.get_shape().as_list()[-1]
                out_shape = tf.shape(tf.tile(x_input, [1, 1, 1, 1, out_filters]))
                x = self.transition_up(x, out_shape, 'TD_out', num_filters=out_filters)
                print('TU_out shape: {}'.format(x.get_shape()))
                x = BN_Relu_conv_3d(x, 3, 256, 'pre_output_layer', add_reg=self.conf.use_reg,
                                    is_train=self.is_training)
                print('pre_out shape: {}'.format(x.get_shape()))
                self.logits = BN_Relu_conv_3d(x, 1, self.conf.num_cls, 'Output_layer',
                                              add_reg=self.conf.use_reg, is_train=self.is_training)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_blocks, scope):
        with tf.name_scope(scope):
            layers_concat = list()
            layers_concat.append(layer_input)
            x = self.bottleneck_block(layer_input, scope=scope + '_BB_' + str(0))
            layers_concat.append(x)
            for i in range(num_blocks - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_block(x, scope=scope + '_BB_' + str(i + 1))
                layers_concat.append(x)
            x = concatenation(layers_concat)
        return x

    def bottleneck_block(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training, scope='BN1')
            x = Relu(x)
            x = conv_3d(x, filter_size=1, num_filters=4 * self.k, layer_name='conv1', add_reg=self.conf.use_reg)
            x = drop_out(x, keep_prob=self.keep_prob)

            x = batch_norm(x, is_training=self.is_training, scope='BN2')
            x = Relu(x)
            x = conv_3d(x, filter_size=3, num_filters=self.k, layer_name='conv2', add_reg=self.conf.use_reg)
            x = drop_out(x, keep_prob=self.keep_prob)
            return x

    def transition_down(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training, scope='BN')
            x = Relu(x)
            x = conv_3d(x, filter_size=1, num_filters=self.k, layer_name='conv', add_reg=self.conf.use_reg)
            x = drop_out(x, keep_prob=self.keep_prob)
            x = avg_pool(x, ksize=2, stride=2, scope='avg_pool')
            return x

    def transition_up(self, x, out_shape, scope, num_filters=None):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training, scope='BN')
            x = Relu(x)
            if not num_filters:
                num_filters = self.k
            x = deconv_3d(inputs=x,
                          filter_size=3,
                          num_filters=num_filters,
                          layer_name='deconv',
                          stride=2,
                          add_reg=self.conf.use_reg,
                          add_batch_norm=False,
                          is_train=self.is_training,
                          out_shape=out_shape)
            x = drop_out(x, keep_prob=self.keep_prob)
        return x
