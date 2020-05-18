import os
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/user/rzd/'


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'rs19_val':
            return os.path.join(ROOT_DIR, 'datasets', 'rs19_val')
        if dataset == 'cityscapes':
            return os.path.join(ROOT_DIR, 'datasets', 'cityscapes')
        if dataset == 'a2d2_audi':
            return os.path.join(ROOT_DIR, 'datasets', 'camera_lidar_semantic')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
