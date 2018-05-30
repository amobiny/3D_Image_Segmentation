import os
from PIL import Image
import numpy as np
import h5py


def read_images(path, img_size=(128, 128, 64)):
    """
    reads all images in a folder one-by-one (in alphabetical order) and converts them to 3D images
    :param path: folder path
    :param img_size: size of final 3D images
    :return: 3D image of size (1, height, width, depth, 1)
    """
    img_3d = np.zeros((img_size[0], img_size[1], 0))
    image_name_list = os.listdir(path)
    image_name_list.sort()
    for img_name in image_name_list:
        filename = path + '/' + img_name
        img = np.array(Image.open(filename))
        if len(img.shape) == 3:     # if image is saved as RGB (i.e. with three similar channels)
            img = img[:, :, 0]
        img_3d = np.concatenate((img_3d, img.reshape(img_size[0], img_size[1], 1)), axis=-1)
    assert (img_3d.shape[-1] == img_size[-1]), "depth of the generated 3D-image is not as specified for {}".format(path)
    return np.expand_dims(np.expand_dims(img_3d, axis=-1), axis=0)


def get_data(path, img_size=(128, 128, 64)):
    """
    loads all images and corresponding masks and converts them to numpy arrays
    :param path: path to the destination folder (which includes raw and mask folders)
    :param img_size: size of the images
    :return: 4D arrays (#images, height, width, depth, 1) of images and corresponding masks
    """
    folders = os.listdir(path)
    folders.sort()
    annotation_folders = folders[:len(folders) / 2]
    input_folders = folders[len(folders) / 2:]
    images = np.zeros([0] + list(img_size) + [1])
    masks = np.zeros([0] + list(img_size) + [1])
    for folder in input_folders:
        f_path = path + folder
        image_3d = read_images(f_path)
        images = np.concatenate((images, image_3d), axis=0)
    for folder in annotation_folders:
        f_path = path + folder
        image_3d = read_images(f_path)
        masks = np.concatenate((masks, image_3d), axis=0)
    return images, np.squeeze(masks)


train_path = './data/train_data/'
x_train, y_train = get_data(train_path)
h5f = h5py.File(train_path + 'train.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('y_train', data=y_train)
h5f.close()

