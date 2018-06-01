import tensorflow as tf
# import tensorlayer as tl
import numpy as np
import random
import scipy.ndimage


def cross_entropy(y, logits, n_class):
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(y, [-1, n_class])
    try:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels))
    except:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    return loss


def dice_coeff(y, logits):
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
    try:
        sum_exp = tf.reduce_sum(exponential_map, 4, keepdims=True)
    except:
        sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    # tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    tensor_sum_exp = tf.tile(sum_exp, (1, 1, 1, 1, num_classes))
    return tf.div(exponential_map, tensor_sum_exp)


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)


def random_rotation_3d(batch, max_angle):
    """
    Randomly rotate an image by a random angle (-max_angle, max_angle)
    :param batch: batch of 3D images
    :param max_angle: `float`. The maximum rotation angle
    :return: A batch of rotated 3D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image1 = np.squeeze(batch[i])
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
            #                print(i)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)
