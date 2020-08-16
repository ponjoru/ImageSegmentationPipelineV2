import numpy as np
import losses.functional as F
from utils._utils import tensors_to_numpy, metrics2str


class Evaluator(object):
    """
    Metrics evaluator class.
    Accumulates per batch metrics on every call of the add_batch function. To reset metrics and batch counter call the
    reset function. To get evaluated metrics call the eval_metrics function. (Metrics divided by the amount of batches
    fed to the evaluator).
    :param metrics: list of metrics to process, one of ['hard_iou', 'hard_dice', 'acc',
                    'fwiou', 'precision', 'recall']
    :param num_class: number of classes C (C for multiclass segmentation, C=1 for binary segmentation)
    :param threshold: threshold for the binary segmentation mode (default: 0.5)
    Shape:
    - input: torch.Tensor(B,C,H,W) - class probabilities at each prediction (between 0 and 1).
    - target: torch.Tensor(B,H,W) - ground truth labels (between 0 and C - 1)
    """
    def __init__(self, metrics, num_class, threshold=0.5):
        self.metrics = metrics
        self.num_class = num_class
        self.threshold = threshold
        self.eps = 1e-8     # epsilon to handle possible divisions by zero
        self.confusion_matrix = 0
        self.metrics_dict = {
            'iou': self._hard_iou,
            'dice': self._hard_dice,
            'acc': self._pixel_accuracy,
            'fwiou': self._frequency_weighted_iou,
            'precision': self._precision,
            'recall': self._recall,
        }
        if any([metric not in self.metrics_dict for metric in self.metrics]):
            raise NotImplementedError
        self.metrics_res = dict.fromkeys(metrics, 0)  # Dict of calculated metrics per each class

    def add_batch(self, input, target):
        """
        Accumulate batch data, generates confusion matrix and accumulates existed metrics by evaluated metrics over the
        given batch
        :param input: (B,C,H,W) - Variable, class probabilities at each prediction (between 0 and 1).
        :param target: (B,H,W) - Tensor, ground truth labels (between 0 and C - 1)
        :return:
        """
        assert input.shape[0] == target.shape[0], 'Batch size mismatch'
        assert input.shape[1] == self.num_class, 'Num classes mismatch'
        assert input.shape[2:] == target.shape[1:], (input.shape[2:], target.shape[1:])

        if self.num_class == 1:
            input = (input.squeeze(1) > self.threshold).int()
        else:
            input = input.max(dim=1)[1]
        input, target = tensors_to_numpy(input, target)
        self.confusion_matrix += self._generate_matrix(input, target)

    def reset(self):
        """
        Resets metrics accumulation during training. Call in the end of each epoch
        :return:
        """
        self.metrics_res = dict.fromkeys(self.metrics, 0.0)
        self.confusion_matrix = 0

    def eval_metrics(self, reduction='mean', show=True):
        """
        Returns metrics values dict
        :param reduction: reduction mode, one of 'mean', 'sum', 'none' (default: mean)
        :param show: whether to print metrics or not (default: True)
        :return: metrics values dict
        """
        for metric in self.metrics:
            self.metrics_res[metric] = self.metrics_dict[metric]()
        for key in self.metrics:
            if reduction == 'mean':
                self.metrics_res[key] = np.nanmean(self.metrics_res[key])
            if reduction == 'sum':
                self.metrics_res[key] = np.sum(self.metrics_res[key])
        if show:
            print(metrics2str(self.metrics_res))
        return self.metrics_res

    def _generate_matrix(self, pre_image, gt_image):
        """
        Generates confusion matrix for a bacth of images
        :param pre_image: (B,C,H,W)
        :param gt_image: (B,H,W)
        :return: confusion matrix CxC (2x2 in binary case)
        """
        eval_num_class = 2 if self.num_class == 1 else self.num_class   # define the size of the confusion matrix
        confusion_matrix = np.zeros((eval_num_class,) * 2)

        mask = (gt_image >= 0) & (gt_image < eval_num_class)
        label = eval_num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=eval_num_class ** 2)
        confusion_matrix += count.reshape(eval_num_class, eval_num_class)
        return confusion_matrix

    def _pixel_accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.num_class == 1:
            return Acc[1]
        return Acc

    def _hard_iou(self):
        if self.num_class == 1:
            intersection = self.confusion_matrix[1][1]
            union = self.confusion_matrix[0][1] + self.confusion_matrix[1][0] + self.confusion_matrix[1][1]
        else:
            intersection = np.diag(self.confusion_matrix)
            union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix,
                                                                   axis=0) - intersection + self.eps
        return intersection / (union + self.eps)

    def _hard_dice(self):
        if self.num_class == 1:
            numerator = 2 * self.confusion_matrix[1][1]
            denominator = 2 * self.confusion_matrix[1][1] + self.confusion_matrix[0][1] + self.confusion_matrix[1][0]
        else:
            numerator = 2 * np.diag(self.confusion_matrix)
            denominator = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) + self.eps
        return numerator / (denominator + 1e-8)

    def _frequency_weighted_iou(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _precision(self):
        if self.num_class == 1:
            return self.confusion_matrix[1, 1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def _recall(self):
        if self.num_class == 1:
            return self.confusion_matrix[1, 1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)


if __name__ == "__main__":
    import torch

    input = torch.from_numpy(np.array([[
        [
            [0.8, 0.0, 0.3],
            [0.6, 0.1, 0.2],
            [0.9, 0.1, 0.1],
        ],
        [
            [0.1, 0.8, 0.2],
            [0.3, 0.8, 0.1],
            [0.1, 0.8, 0.0],
        ],
        [
            [0.1, 0.2, 0.5],
            [0.1, 0.1, 0.7],
            [0.0, 0.1, 0.9],
        ],
    ]])).float()
    target = torch.from_numpy(np.array([
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
    ])).long()

    input_bin = torch.from_numpy(np.array([[
        [
            [0.8, 0.2, 0.1],
            [0.2, 0.8, 0.2],
            [0.1, 0.2, 0.8],
        ]
    ]])).float()
    target_bin = torch.from_numpy(np.array([
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ])).long()

    metrics = ['iou', 'dice', 'acc', 'fwiou', 'precision', 'recall']

    print('Binary metrics')
    e = Evaluator(metrics=metrics,  num_class=input_bin.size()[1], threshold=0.5)
    e.add_batch(input_bin, target_bin)
    metrics = e.eval_metrics(reduction='none', show=True)

    print('Multiclass metrics')
    e = Evaluator(metrics=metrics,  num_class=input.size()[1], threshold=0.5)
    e.add_batch(input, target)
    metrics = e.eval_metrics(reduction='none', show=True)
