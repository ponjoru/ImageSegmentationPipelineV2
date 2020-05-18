import numpy as np
import losses.functional as F
from utils.utils_test import tensors_to_numpy


class Evaluator(object):
    """
    Metrics evaluator class.
    Accumulates per batch metrics on every call of the add_batch function. To reset metrics and batch counter call the
    reset function. To get evaluated metrics call the eval_metrics function. (Metrics divided by the amount of batches
    fed to the evaluator).
    :param metrics: list of metrics to process, one of ['soft_iou', 'soft_dice', 'hard_iou', 'hard_dice', 'acc',
                    'fwiou', 'precision', 'recall']
    :param num_class: number of classes C (C for multiclass segmentation, C=1 for binary segmentation)
    :param threshold: threshold for the binary segmentation mode (default: 0.5)
    :param class_weights:
    Shape:
    - input: torch.Tensor(B,C,H,W) - class probabilities at each prediction (between 0 and 1).
    - target: torch.Tensor(B,H,W) - ground truth labels (between 0 and C - 1)
    """
    def __init__(self, metrics, num_class, threshold=0.5, class_weights=None, cuda=False):
        self.metrics = metrics
        self.num_class = num_class
        self.threshold = threshold
        self.class_weights = class_weights
        self.cuda = cuda
        self.eps = 1e-8     # epsilon to handle possible divisions by zero
        self.counter = 0    # current amount of batches fed
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.metrics_dict = {
            'soft_iou': self._soft_jaccard,
            'soft_dice': self._soft_dice,
            'hard_iou': self._hard_iou,
            'hard_dice': self._hard_dice,
            'acc': self._pixel_accuracy,
            'fwiou': self._frequency_weighted_iou,
            'precision': self._precision,
            'recall': self._recall,
        }
        self.metrics_res = dict.fromkeys(self.metrics_dict.keys(), 0)   # Dict of calculated metrics per each class
        if any([metric not in self.metrics_dict for metric in self.metrics]):
            raise NotImplementedError

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

        self.confusion_matrix = self._generate_matrix(input, target)
        for metric in self.metrics:
            self.metrics_res[metric] += self.metrics_dict[metric](input, target)

        self.counter += 1

    def reset(self):
        """
        Resets metrics accumulation during training. Call in the end of each epoch
        :return:
        """
        self.counter = 0
        self.metrics_res = dict.fromkeys(self.metrics_dict.keys(), 0.0)
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def eval_metrics(self, reduction='mean', show=True):
        """
        Returns metrics values dict
        :param reduction: reduction mode, one of 'mean', 'sum', 'none' (default: mean)
        :param show: whether to print metrics or not (default: True)
        :return: metrics values dict
        """
        assert self.counter > 0, 'Call add_batch first'
        results = []
        for key in self.metrics:
            if reduction == 'mean':
                self.metrics_res[key] = np.nanmean(self.metrics_res[key]) / self.counter
            if reduction == 'sum':
                self.metrics_res[key] = np.sum(self.metrics_res[key]) / self.counter
            if reduction == 'none':
                s = '{'
                for ind, value in enumerate(self.metrics_res[key] / self.counter):
                    s += '%i: %1.4f, ' % (ind, value)
                results.append('%s: %s' % (key, s[:-2] + '}'))
            else:
                results.append('%s: %1.4f' % (key, float(self.metrics_res[key])))
        if show and reduction == 'none':
            print(*results, sep='\n')
        if show and reduction != 'none':
            print(results, sep=', ')
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
        gt_image = gt_image.cpu().detach().numpy()
        pre_image = pre_image.cpu().detach().numpy()
        for i in range(gt_image.shape[0]):
            gt_mask = gt_image[i]

            if eval_num_class == 2:
                pre_mask = pre_image.squeeze(axis=1)[i] > self.threshold
            else:
                pre_mask = np.argmax(pre_image[i], axis=0)

            mask = (gt_mask >= 0) & (gt_mask < eval_num_class)
            label = eval_num_class * gt_mask[mask].astype('int') + pre_mask[mask]
            count = np.bincount(label, minlength=eval_num_class ** 2)
            confusion_matrix += count.reshape(eval_num_class, eval_num_class)
        return confusion_matrix

    def _pixel_accuracy(self, input, target):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.num_class == 1:
            return Acc[1]
        return Acc

    def _soft_jaccard(self, input, target):
        tmp = F.soft_jaccard_index(input, target)
        return tensors_to_numpy(tmp, cuda=self.cuda)[0]

    def _soft_dice(self, input, target):
        tmp = F.soft_dice_index(input, target)
        return tensors_to_numpy(tmp, cuda=self.cuda)[0]

    def _hard_iou(self, input, target):
        if self.num_class == 1:
            intersection = self.confusion_matrix[1][1]
            union = self.confusion_matrix[0][1] + self.confusion_matrix[1][0] + self.confusion_matrix[1][1]
        else:
            intersection = np.diag(self.confusion_matrix)
            union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix,
                                                                   axis=0) - intersection + self.eps
        return intersection / (union + self.eps)

    def _hard_dice(self, input, target):
        if self.num_class == 1:
            numerator = 2 * self.confusion_matrix[1][1]
            denominator = 2 * self.confusion_matrix[1][1] + self.confusion_matrix[0][1] + self.confusion_matrix[1][0]
        else:
            numerator = 2 * np.diag(self.confusion_matrix)
            denominator = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) + self.eps
        return numerator / (denominator + 1e-8)

    def _frequency_weighted_iou(self, input, target):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _precision(self, input, target):
        if self.num_class == 1:
            return self.confusion_matrix[1, 1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def _recall(self, input, target):
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

    metrics = ['soft_iou', 'soft_dice', 'hard_iou', 'hard_dice', 'acc', 'fwiou', 'precision', 'recall']

    print('Binary metrics')
    e = Evaluator(metrics=metrics,  num_class=input_bin.size()[1], threshold=0.5)
    e.add_batch(input_bin, target_bin)
    metrics = e.eval_metrics(reduction='mean', show=True)

    print('Multiclass metrics')
    e = Evaluator(metrics=metrics,  num_class=input.size()[1], threshold=0.5)
    e.add_batch(input, target)
    metrics = e.eval_metrics(reduction='mean', show=True)
