import numpy as np
from torch.utils import data
from abc import ABC, abstractmethod
from PIL import Image
from utils.data_plotters.SegmentationPlotter import SegmentationPlotter


class ISDatasetTemplate(data.Dataset, ABC):
    """
    Template of Image segmentation Dataset
    Must be overridden:
        def __len__(self)
        def __getitem__(self, index)
        def transform_tr(self, sample)
        def transform_val(self, sample)
        def transform_ts(self, sample)
    :return: sample = {'image': tensor(3,H,W), 'label': tensor(H,W), 'id': str(image name)}
    """
    def __init__(self, labels, binary_segmentation, threshold=0.5):
        self.labels = sorted(labels, key=lambda x: x.train_id)
        self.threshold = threshold
        self.binary_segmentation = binary_segmentation
        if self.binary_segmentation:
            self.num_classes = 1
        else:
            self.num_classes = max(self.labels, key=lambda x: x.train_id if x.train_id != 255 else 0).train_id + 1
        self.transformations = {
            'train': self.transform_tr,
            'val': self.transform_val,
            'test': self.transform_ts,
        }
        self.plotter = SegmentationPlotter(self.labels, self.num_classes, self.threshold)

    @abstractmethod
    def __len__(self):
        """
        Returns the size of the loader in batches
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns a single item from dataset by index
        :param index:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def transform_tr(self, sample):
        """
        Transformations over the training data
        :param sample:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def transform_val(self, sample):
        """
        Transformations over the validation data
        :param sample:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def transform_ts(self, sample):
        """
        Transformations over the test data
        :param sample:
        :return:
        """
        raise NotImplementedError

    def map_target(self, target):
        """
        Maps target encoded with classic id classes to target encoded in train_id classes
        """
        target = np.array(target, dtype=np.uint8)
        map_tgt = np.zeros_like(target)
        for label in self.labels:
            map_tgt[target == label.id] = label.train_id
        return Image.fromarray(map_tgt)
