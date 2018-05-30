import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')

flags.DEFINE_integer('max_step', 250000, '# of step for training')
flags.DEFINE_integer('test_interval', 100000000, '# of interval to test a model')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 1000, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', 1e-3, 'learning rate')
flags.DEFINE_float('lmbda', 1e-3, 'L2 regularization coefficient')

# data
flags.DEFINE_string('data_dir', '../h5_data_SA/', 'Name of data file(s)')
flags.DEFINE_boolean('aug_flip', False, 'Training data augmentation: flip. Extra 3 datasets.')
flags.DEFINE_boolean('aug_rotate', False, 'Training data augmentation: rotate. Extra 9 datasets.')
flags.DEFINE_integer('patch_size', 32, 'patch size')
flags.DEFINE_integer('overlap_stepsize', 16, 'overlap stepsize when performing testing')
flags.DEFINE_integer('batch_size', 5, 'training batch size')
flags.DEFINE_integer('channel', 1, 'channel size')
flags.DEFINE_integer('depth', 32, 'depth size')     # should be equal to patch_size
flags.DEFINE_integer('height', 32, 'height size')   # should be equal to patch_size
flags.DEFINE_integer('width', 32, 'width size')     # should be equal to patch_size
# Debug
flags.DEFINE_string('logdir', './log_dir', 'Logs directory')
flags.DEFINE_string('modeldir', './model_dir', 'Model directory')
flags.DEFINE_string('savedir', './result', 'Result saving directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_integer('test_step', 150000, 'Test or predict model at this step')
flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
# network architecture
flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
flags.DEFINE_integer('num_cls', 2, 'Number of output classes')
flags.DEFINE_integer('start_channel_num', 16, 'start number of outputs for the first conv layer')
flags.DEFINE_integer('filter_size', 3, 'Filter size for the conv and deconv layers')
flags.DEFINE_integer('pool_filter_size', 2, 'Filter size for pooling layers')

args = tf.app.flags.FLAGS
