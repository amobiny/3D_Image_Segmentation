import numpy as np
import h5py


class DataLoader(object):

    def __init__(self, cfg):
        self.train_data_dir = cfg.train_data_dir
        self.valid_data_dir = cfg.valid_data_dir
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        self.height, self.width, self.depth = cfg.height, cfg.width, cfg.depth
        self.max_bottom_left_front_corner = (cfg.height - 1, cfg.width - 1, cfg.depth - 1)
        # maximum value that the bottom left front corner of a cropped patch can take

    def next_batch(self):
        img_idx = np.sort(np.random.choice(self.num_tr, replace=False, size=self.batch_size))
        bottom = np.random.randint(self.max_bottom_left_front_corner[1])
        left = np.random.randint(self.max_bottom_left_front_corner[0])
        front = np.random.randint(self.max_bottom_left_front_corner[2])
        h5f = h5py.File(self.train_data_dir + 'train.h5', 'r')
        x = h5f['x_train'][img_idx, bottom:bottom+self.height, left:left+self.width, front:front+self.depth, :]
        y = h5f['y_train'][img_idx, bottom:bottom+self.height, left:left+self.width, front:front+self.depth]
        h5f.close()
        return x, y

    def get_validation(self):
        h5f = h5py.File(self.valid_data_dir + 'valid.h5', 'r')
        x = h5f['x_valid'][:]
        y = h5f['y_valid'][:]
        h5f.close()
        return x, y
