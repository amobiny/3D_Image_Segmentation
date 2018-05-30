import os
import tensorflow as tf
from ops import conv_3d, max_pool, deconv_3d


class Unet_3D(object):
    # Class properties
    __network = None         # Graph for UNET
    __train_op = None        # Operation used to optimize loss function
    __loss = None            # Loss function to be optimized, which is based on predictions
    __accuracy = None        # Classification accuracy for all conditions
    __probs = None           # Prediction probability matrix of shape [batch_size, numClasses]
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.k_size = self.conf.filter_size
        self.pool_size = self.conf.pool_filter_size
        self.input_shape = [self.conf.batch_size, self.conf.depth, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [self.conf.batch_size, self.conf.depth, self.conf.height, self.conf.width]
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.savedir):
            os.makedirs(conf.savedir)
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.keep_prob = tf.placeholder(tf.float32)

    def inference(self):
        if self.__network:
            return self
        # Building network...
        with tf.variable_scope('AlexNet'):
            conv1 = conv_3d(self.x, self.k_size, 16, 'CONV1')
            conv2 = conv_3d(conv1, self.k_size, 32, 'CONV2')
            pool1 = max_pool(conv2, self.pool_size, 'MaxPool1')
            deconv1 = deconv_3d(pool1, self.k_size, 16, 'DECONV1')
            merge1 = tf.concat([conv2, deconv1], -1, name='concat')
            self.__logits = conv_3d(merge1, self.k_size, self.conf.num_cls, 'CONV3')
        return self

    def loss_func(self):
        if self.__loss:
            return self
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.y, depth=self.conf.num_cls, axis=4, name='y_one_hot')
            with tf.name_scope('cross_entropy'):
                losses = tf.losses.softmax_cross_entropy(y_one_hot, self.__logits, scope='losses')
                cross_entropy = tf.reduce_mean(losses, name='loss')
                tf.summary.scalar('cross_entropy', cross_entropy)
            with tf.name_scope('L2_loss'):
                l2_loss = tf.reduce_sum(
                    self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
                tf.summary.scalar('l2_loss', l2_loss)
            with tf.name_scope('total'):
                self.__loss = cross_entropy + l2_loss
        return self





    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.__loss))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summary = tf.summary.merge(summarys)
        return summary



    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)



    @property
    def network(self):
        return self.__network















