import tensorflow as tf
from Data_Loader import DataLoader
from ops import conv_3d, max_pool, deconv_3d
from utils import cross_entropy, dice_coeff
import os


class Unet_3D(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.k_size = self.conf.filter_size
        self.pool_size = self.conf.pool_filter_size
        self.input_shape = [self.conf.batch_size, self.conf.height, self.conf.width, self.conf.depth, self.conf.channel]
        self.output_shape = [self.conf.batch_size, self.conf.height, self.conf.width, self.conf.depth]
        self.create_placeholders()
        self.configure_network()
        self.configure_summary()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.is_training = True
            # self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
            # self.keep_prob = tf.placeholder(tf.float32)

    def inference(self):
        # Building network...
        with tf.variable_scope('3D_UNET'):
            conv1 = conv_3d(self.x, self.k_size, 16, 'CONV1', is_train=self.is_training)
            conv2 = conv_3d(conv1, self.k_size, 32, 'CONV2', is_train=self.is_training)
            pool1 = max_pool(conv2, self.pool_size, 'MaxPool1')
            deconv1 = deconv_3d(pool1, self.k_size, 16, 'DECONV1', is_train=self.is_training)
            merge1 = tf.concat([conv2, deconv1], -1, name='concat')
            self.logits = conv_3d(merge1, self.k_size, self.conf.num_cls, 'CONV3', is_train=self.is_training)
            self.loss_func()
            self.accuracy_func()

    def loss_func(self):
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.y, depth=self.conf.num_cls, axis=4, name='y_one_hot')
            if self.conf.loss_type == 'cross-entropy':
                with tf.name_scope('cross_entropy'):
                    loss = cross_entropy(y_one_hot, self.logits, self.conf.num_cls)
            elif self.conf.loss_type == 'dice':
                with tf.name_scope('dice_coefficient'):
                    loss = dice_coeff(y_one_hot, self.logits)
            with tf.name_scope('L2_loss'):
                l2_loss = tf.reduce_sum(
                    self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
            with tf.name_scope('total'):
                self.loss = loss + l2_loss

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            y_pred = tf.argmax(self.logits, axis=4, name='decode_pred')
            correct_prediction = tf.equal(self.y, y_pred, name='correct_pred')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')

    def configure_network(self):
        self.inference()
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.init_lr)
            self.train_op = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('train/loss', self.loss),
                        tf.summary.scalar('train/accuracy', self.accuracy),
                        tf.summary.scalar('valid/loss', self.loss),
                        tf.summary.scalar('valid/accuracy', self.accuracy)]
        self.train_summary = tf.summary.merge(summary_list[:2])
        self.valid_summary = tf.summary.merge(summary_list[2:])

    def save_summary(self, summary, step):
        print('----> Summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        print('Start Training')
        data_reader = DataLoader(self.conf)
        for train_step in range(1, self.conf.max_step+1):
            print('Step: {}'.format(train_step))
            self.is_training = True
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
                self.is_training = False
                x_val, y_val = data_reader.get_validation()
                feed_dict = {self.x: x_val, self.y: y_val}
                loss, acc, summary = self.sess.run([self.loss, self.accuracy, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, train_step+self.conf.reload_step)
                print('-'*30+'Validation'+'-'*30)
                print('step: {0:<6}, val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step, loss, acc))
            if train_step % self.conf.SAVE_FREQ == 0:
                self.save(train_step+self.conf.reload_step)

    def test(self):
        pass

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

