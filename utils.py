import tensorflow as tf
import tensorlayer as tl


def cross_entropy(y, logits, n_class):
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(y, [-1, n_class])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    return loss


def dice_coeff(y, logits, n_class):
    eps = 1e-5
    prediction = pixel_wise_softmax(logits)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    dice_loss = 1 - (2 * intersection / union)
    # outputs = tl.act.pixel_wise_softmax(logits)
    # dice_loss = 1 - tl.cost.dice_coe(outputs, y, loss_type='jaccard', axis=(1, 2, 3, 4))
    return dice_loss


def pixel_wise_softmax(output_map):
    num_classes = output_map.get_shape().as_list()[-1]
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    # tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    tensor_sum_exp = tf.tile(sum_exp, (1, 1, 1, 1, num_classes))
    return tf.div(exponential_map, tensor_sum_exp)