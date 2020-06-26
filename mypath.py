import os
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/your/working/dir'


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return os.path.join(ROOT_DIR, 'datasets', 'cityscapes')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
