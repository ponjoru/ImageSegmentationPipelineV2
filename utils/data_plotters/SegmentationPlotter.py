import os
import time
import numpy as np
from matplotlib.axes import Axes
from utils.utils_test import create_canvas


class SegmentationPlotter:
    """
    Provides API for plotting semantic segmentation output and ground truth data
    """
    def __init__(self, labels, num_classes, threshold):
        self.labels = labels
        self.num_classes = num_classes
        self.threshold = threshold
        self.binary_segmentation = num_classes == 1

    def _standard_encoding_mask_to_rgb(self, mask, threshold=None):
        """
        :param mask: np.array(H, W), where each item: 0 <= item <= C-1
        :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default 1)
        :return: rgb_mask: np.array(H,W,3)
        """
        if threshold is None:
            threshold = 1
        height, width = mask.shape
        rgb_mask = np.zeros((height, width, 3))
        if self.binary_segmentation:
            mask[mask == 255] = 0
            rgb_mask[mask >= threshold] = self.labels[0].color
        else:
            for i in range(self.num_classes):
                rgb_mask[mask == i] = self.labels[i].color
        rgb_mask = rgb_mask.astype(float) / 255
        return rgb_mask

    def _one_hot_encoding_mask_to_rgb(self, mask, threshold=None):
        """
        :param mask: one-hot-encoded mask = np.array(C, H, W), where each cell = 0 or 1
        :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default self.threshold)
        :return: rgb_mask: np.array(H,W,3)
        """
        if threshold is None:
            threshold = self.threshold

        c, height, width = mask.shape
        binary = c == 1
        # translate mask from one-hot to standard encoding
        if binary:
            mask = mask.squeeze()
        else:
            zeros = np.all(mask[:,:,:] == 0, axis=0)
            mask = mask.argmax(axis=0)
            mask[zeros] = 255
        return self._standard_encoding_mask_to_rgb(mask, threshold=threshold)

    def plot_result(self, _img, _target, _ground_truth, alpha=0.4, threshold=None, show=False):
        """
        Builds 2 subplots with: img covered by target with alpha, img covered by ground_truth with alpha
        :param _img: np.array(3,W,H)
        :param _target: np.array(C,W,H)
        :param _ground_truth: np.array(W,H)
        :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque) (default: 0.4)
        :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default: None)
        :param show: whether to show the image (default: False)
        :return:
        """
        fig, (ax_target, ax_ground_truth) = create_canvas(nrows=1, ncols=2, a=1.3, b=2.5, image_size=_img.shape[1:])
        ax_target.set_title('Prediction')
        ax_ground_truth.set_title('Ground truth')
        self.plot_inference(ax_target, _img, _target, threshold=threshold)
        self.plot_ground_truth(ax_ground_truth, _img, _ground_truth, alpha=alpha)
        if show:
            fig.show()
        return fig

    def plot_inference(self, axis, _img, _target, alpha=0.4, threshold=0.5):
        """
        Plots image covered by predicted segmentation mask in a single plot
        :param _img: np.array(3,W,H)
        :param _target: np.array(C,W,H)
        :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque) (default: 0.4)
        :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default: 0.5)
        :return:
        """
        assert isinstance(axis, Axes)
        img = np.transpose(_img, (1, 2, 0))
        rgb_target = self._one_hot_encoding_mask_to_rgb(_target, threshold=threshold)

        axis.imshow(img)
        axis.imshow(rgb_target, alpha=alpha)

    def plot_ground_truth(self, axis, _img, _gt, alpha=0.4):
        """
        Plots image covered by ground truth segmentation mask in a single plot
        :param _img: np.array(3,W,H)
        :param _gt: np.array(W,H)
        :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque) (default: 0.4)
        :return:
        """
        assert isinstance(axis, Axes)
        img = np.transpose(_img, (1, 2, 0))
        rgb_gt = self._standard_encoding_mask_to_rgb(_gt)

        axis.imshow(img)
        axis.imshow(rgb_gt, alpha=alpha)
