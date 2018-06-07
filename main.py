import tensorflow as tf
import argparse
from config import args
from model.UNet import UNET_3D
from model.FCNet import FCN_3D
from model.VNet import VNet

import os


def main(_):
    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = VNet(tf.Session(), args)
        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    tf.app.run()
