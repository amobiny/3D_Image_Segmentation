

class DataLoader(object):
    def __init__(self, cfg):
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.batch_size

    def next_batch(self):
        pass

    def get_validation(self):
        pass
