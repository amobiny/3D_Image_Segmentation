import os
import tensorflow as tf
from ops import conv_3d, max_pool, deconv_3d


class Unet_3D(object):
    # Class properties
    __logits = None         # Graph for UNET
    __train_op = None       # Operation used to optimize loss function
    __loss = None           # Loss function to minimize
    __accuracy = None       # Segmentation accuracy
    __train_summary = None  # Train summaries
    __valid_summary = None  # Validation summaries

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
        if self.__logits:
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

    def accuracy_func(self):
        if self.__accuracy:
            return self
        with tf.name_scope('Accuracy'):
            y_pred = tf.argmax(self.__logits, axis=4, name='decode_pred')
            correct_prediction = tf.equal(self.y, y_pred, name='correct_pred')
            self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
        return self

    def configure_network(self):
        if self.__train_op:
            return self
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        return self

    def config_summary(self):
        if self.__train_summary and self.__valid_summary:
            return self
        summary_list = [tf.summary.scalar('train/loss', self.__loss),
                        tf.summary.scalar('train/accuracy', self.__accuracy),
                        tf.summary.scalar('valid/loss', self.__loss),
                        tf.summary.scalar('valid/accuracy', self.__accuracy)]
        self.__train_summary = tf.summary.merge(summary_list[:2])
        self.__valid_summary = tf.summary.merge(summary_list[2:])
        return self

    def save_summary(self, summary, step):
        print('---->Summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        data_reader = DataLoader(self.conf.data_dir)
        for train_step in range(1, self.conf.max_step+1):

            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = data_reader.next_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch}
                _, loss, acc, summary = self.sess.run([self.train_op, self.loss, self.accuracy, self.train_summary],
                                                      feed_dict=feed_dict)
                self.save_summary(summary, train_step+self.conf.reload_step)
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
            else:
                x_batch, y_batch = data_reader.next_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch}
                self.sess.run(self.train_op, feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                x_val, y_val = data_reader.get_validation()
                feed_dict = {self.x: x_val, self.y: y_val}
                loss, acc, summary = self.sess.run([self.loss, self.accuracy, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, train_step+self.conf.reload_step)
                print('-'*30+'Validation'+'-'*30)
                print('step: {0:<6}, val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step, loss, acc))
            if train_step % self.conf.SAVE_FREQ == 0:
                self.save(train_step+self.conf.reload_step)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('Model successfully restored')

    @property
    def network(self):
        return self.__logits

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def train_summary(self):
        return self.__train_summary

    @property
    def valid_summary(self):
        return self.__valid_summary










