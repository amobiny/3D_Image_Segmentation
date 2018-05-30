import os
from PIL import Image
import numpy as np
import cv2
import glob


def read_images(path, img_size=(128, 128, 64)):
    img_3d = np.zeros((img_size[0], img_size[1], 0))
    for filename in glob.glob(path + '/*.bmp'):
        img = np.array(Image.open(filename))
        if len(img.shape) == 3:     # if image is saved as RGB (i.e. with three similar channels)
            img = img[:, :, 0]
        img_3d = np.concatenate((img_3d, img.reshape(img_size[0], img_size[1], 1)), axis=-1)
    assert (img_3d.shape[-1] == img_size[-1]), "depth of the generated 3D-image is not as specified for {}".format(path)
    return img_3d


def get_data(path, img_size=(128, 128, 64)):
    folders = os.listdir(path)
    folders.sort()
    annotation_folders = folders[:len(folders) / 2]
    input_folders = folders[len(folders) / 2:]
    images = np.zeros([0] + list(img_size))
    masks = np.zeros([0] + list(img_size))
    for folder in input_folders:
        f_path = path + folder
        image_3d = read_images(f_path)
        images = np.concatenate((images, image_3d.reshape([1] + list(img_size))), axis=0)
    for folder in annotation_folders:
        f_path = path + folder
        image_3d = read_images(f_path)
        masks = np.concatenate((masks, image_3d.reshape([1] + list(img_size))), axis=0)
    return images, masks


train_input, train_mask = get_data('./data/train_data/')

print()
