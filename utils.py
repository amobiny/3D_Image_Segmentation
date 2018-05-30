import tensorflow as tf


def cross_entropy(y, logits, n_class):
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(y, [-1, n_class])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    return loss


def dice_coeff(y, logits, n_class):
    eps = 1e-5
    prediction = pixel_wise_softmax_2(logits)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    loss = -(2 * intersection / union)
    return loss