import tensorflow as tf


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_3d(inputs, filter_size, num_filters, layer_name, is_train=True,
            batch_norm=False, add_reg=False, use_relu=True):
    """
    Create a 3D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    num_in_channel = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        shape = [filter_size, filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        tf.summary.histogram('W', weights)
        # biases = bias_variable(layer_name, [num_filters])
        layer = tf.nn.conv3d(input=inputs,
                             filter=weights,
                             strides=[1, 1, 1, 1, 1],
                             padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        layer = batch_norm_wrapper(layer, is_train)
        # layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def deconv_3d(inputs, filter_size, num_filters, layer_name, is_train=True, add_reg=False, use_relu=True):
    """
    Create a 3D transposed-convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(layer_name):
        kernel_shape = [filter_size, filter_size, filter_size, num_filters, input_shape[-1]]
        out_shape = [input_shape[0]] + list(map(lambda x: x*2, input_shape[1:-1])) + [num_filters]
        weights = weight_variable(layer_name, shape=kernel_shape)
        # biases = bias_variable(layer_name, [num_filters])
        layer = tf.nn.conv3d_transpose(inputs,
                                       filter=weights,
                                       output_shape=out_shape,
                                       strides=[1, 2, 2, 2, 1],
                                       padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        layer = batch_norm_wrapper(layer, is_train)
        # layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def max_pool(x, ksize, name):
    """
    Create a 3D max-pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    maxpool = tf.nn.max_pool3d(x,
                               ksize=[1, ksize, ksize, ksize, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding="SAME",
                               name=name)
    print('{}: {}'.format(maxpool.name, maxpool.get_shape()))
    return maxpool


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    """
    creates a batch normalization layer
    :param inputs: input array
    :param is_training: boolean for differentiating train and test
    :param decay:
    :param epsilon:
    :return: normalized input
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 5:  # For 3D convolutional layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2, 3])
        else:  # For fully-connected layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
