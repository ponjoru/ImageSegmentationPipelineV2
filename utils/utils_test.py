import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def metrics2str(metrics_dict: dict):
    res_str = ''
    for key in metrics_dict.keys():
        res_str += '{:13s}'.format(key)
        # TODO: refactor
        if key in ['iou', 'dice']:
            res_str += ': [mean: %.4f; ' % metrics_dict[key].mean()
            for id, item in enumerate(metrics_dict[key]):
                res_str += '%i: %.4f ' % (id, item)
            res_str += ']\n'
        else:
            res_str += ': %.4f\n' % (metrics_dict[key])
    return res_str


def create_canvas(nrows, ncols, a, b, image_size, dpi=150):
    h, w = image_size
    fig_size = b * w / float(dpi), a * h / float(dpi)
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    return fig, axes


def tensors_to_numpy(*tensors):
    res = []
    for tensor in tensors:
        if tensor.is_cuda:
            res.append(tensor.cpu().detach().numpy())
        else:
            res.append(tensor.detach().numpy())
    return res


def recursive_glob(rootdir='.', suffix='', exclude: list = None):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    res = []
    for looproot, dirs, filenames in os.walk(rootdir, topdown=True):
        if exclude:
            dirs[:] = [d for d in dirs if d not in exclude]
        for filename in filenames:
            if filename.endswith(suffix):
                res.append(os.path.join(looproot, filename))
    return res


def one_hot_encode(input, num_classes):
    """new_input[i, j]
    Calculates one-hot encoding of given image
    :param input: (B,W,H), where each element el: int: 0<= el <= C-1
    :param num_classes: C
    :return: matrix of shape: (B,C,H,W)
    """
    b, h, w = input.shape
    new_input = np.zeros((b, num_classes, h, w))
    for i in range(b):
        if num_classes == 1:
            new_input[i] = input[i, :, :]
        else:
            for j in range(num_classes):
                new_input[i, j][input[i] == j] = 1

    return new_input


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def plot_confusion_matrix(matrix,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          show=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    assert matrix.shape[0] == matrix.shape[1]
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = create_canvas(1, 1, a=1, b=1, image_size=(2048, 2048))
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.0f'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center", fontsize=11,
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _standard_encoding_mask_to_rgb(mask, threshold=None):
    """
    :param mask: np.array(H, W), where each item: 0 <= item <= C-1
    :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default 1)
    :return: rgb_mask: np.array(H,W,3)
    """
    if threshold is None:
        threshold = 1
    height, width = mask.shape
    rgb_mask = np.zeros((height, width, 3))
    rgb_mask[:, :, 0] = mask
    rgb_mask = (rgb_mask >= threshold).astype(float)
    return rgb_mask


def _one_hot_encoding_mask_to_rgb(mask, threshold=None):
    """
    :param mask: one-hot-encoded mask = np.array(C, H, W), where each cell = 0 or 1
    :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default self.threshold)
    :return: rgb_mask: np.array(H,W,3)
    """

    # translate mask from one-hot to standard encoding
    mask = mask.squeeze()
    return _standard_encoding_mask_to_rgb(mask, threshold=threshold)


def plot_result(_img, _target, _ground_truth, alpha=0.4, threshold=None, show=False, save=False, img_name=None):
    """
    Builds 2 subplots with: img covered by target with alpha, img covered by ground_truth with alpha
    :param _img: np.array(3,W,H)
    :param _target: np.array(C,W,H)
    :param _ground_truth: np.array(W,H)
    :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque) (default: 0.4)
    :param threshold: threshold for binary segmentation (target = 1 if target >= threshold else 0) (default: None)
    :param show: whether to show the image (default: False)
    :param save: whether to save the image (default: False)
    :param img_name: image name to save with (default: None (timestamp))
    :return:
    """
    fig, (ax_target, ax_ground_truth) = create_canvas(nrows=1, ncols=2, a=1.3, b=2.5, image_size=_img.shape[1:])
    ax_target.set_title('Prediction')
    ax_ground_truth.set_title('Ground truth')
    plot_inference(ax_target, _img, _target, threshold=threshold)
    plot_ground_truth(ax_ground_truth, _img, _ground_truth, alpha=alpha)
    if save:
        if img_name is None:
            img_name = str(time.strftime("%Y%m%d-%H%M%S"))
        img_name = os.path.join('./results', img_name)
        fig.savefig(img_name, dpi=fig.dpi)
    if show:
        fig.show()
    return fig


def plot_inference(axis, _img, _target, alpha=0.4, threshold=0.5):
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
    rgb_target = _one_hot_encoding_mask_to_rgb(_target, threshold=threshold)

    axis.imshow(img)
    axis.imshow(rgb_target, alpha=alpha)


def plot_ground_truth(axis, _img, _gt, alpha=0.4):
    """
    Plots image covered by ground truth segmentation mask in a single plot
    :param _img: np.array(3,W,H)
    :param _gt: np.array(W,H)
    :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque) (default: 0.4)
    :return:
    """
    assert isinstance(axis, Axes)
    img = np.transpose(_img, (1, 2, 0))
    rgb_gt = _standard_encoding_mask_to_rgb(_gt)

    axis.imshow(img)
    axis.imshow(rgb_gt, alpha=alpha)