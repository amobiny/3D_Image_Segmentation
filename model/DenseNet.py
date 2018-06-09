import tensorflow as tf
from base_model import BaseModel
from ops import conv_3d


class DenseNet(BaseModel):
    def __init__(self, sess, conf):
        # BaseModel.__init__(self, sess, conf)
        super(FCN, self).__init__(sess, conf)
