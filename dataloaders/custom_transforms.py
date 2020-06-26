import torch
import random
import numpy as np
from PIL.Image import Image
import torchvision.transforms as tr
import torchvision.transforms.functional as G


def denormalize_image(tensor: torch.Tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor.clamp(0, 1)


class OneOf(object):
    def __init__(self, *transforms_list):
        self.transforms = transforms_list

    def __call__(self, sample):
        aug = random.choice(self.transforms)
        return aug(sample)


class RandomGaussianNoise(object):
    def __init__(self, p, mean, std, apply_to):
        self.p = p
        self.mean = mean
        self.std = std
        self.apply_to = apply_to

    def __call__(self, sample):
        p = random.random()
        if p > self.p:
            for key in self.apply_to:
                if key not in sample.keys():
                    raise NotImplementedError
                item = sample[key]
                if isinstance(item, torch.Tensor):
                    sample[key] = item + item.normal_(mean=self.mean, std=self.std)
                elif isinstance(item, (np.ndarray, np.generic)):
                    sample[key] = item + np.random.normal(self.mean, self.std, size=item.shape)
                elif isinstance(item, str):
                    pass
                else:
                    raise NotImplementedError(type(item) + ' is not supported')
        return sample


class Crop(object):
    def __init__(self, left=0, upper=0, right=1, lower=1, apply_to=None):
        self.left = left
        self.upper = upper
        self.right = right
        self.lower = lower
        self.apply_to = apply_to

    def __call__(self, sample: dict):
        keys = self.apply_to if self.apply_to else sample.keys()
        for key in keys:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            if isinstance(item, Image):
                w, h = item.size
                sample[key] = item.crop((w * self.left, h * self.upper, w * self.right, h * self.lower))
            elif isinstance(item, (np.ndarray, np.generic)):
                w, h = item.shape[-2:]
                sample[key] = item[:, int(h * self.upper):int(h * self.lower), int(w * self.left):int(w * self.right)]
            elif isinstance(item, str):
                pass
            else:
                raise NotImplementedError(type(item) + ' is not supported')
        return sample


class RandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False, apply_to=None):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.apply_to = apply_to

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        # assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        # if self.padding > 0:
        #     img = F.pad(img, self.padding)
        #     lbl = F.pad(lbl, self.padding)
        #
        # # pad the width if needed
        # if self.pad_if_needed and img.size[0] < self.size[1]:
        #     img = G.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
        #     lbl = G.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))
        #
        # # pad the height if needed
        # if self.pad_if_needed and img.size[1] < self.size[0]:
        #     img = G.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
        #     lbl = G.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        img = sample['image']
        i, j, h, w = self.get_params(img, self.size)
        keys = self.apply_to if self.apply_to else sample.keys()
        for key in keys:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            if isinstance(item, Image):
                sample[key] = G.crop(item, i, j, h, w)
            elif isinstance(item, (np.ndarray, np.generic)):
                sample[key] = sample[key][:, int(i):int(h + i), int(j):int(w + j)]
            elif isinstance(item, str):
                pass
            else:
                raise NotImplementedError(type(item) + ' is not supported')
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class Resize(object):
    def __init__(self, size, apply_to=None):
        self.size = size
        self.apply_to = apply_to

    def __call__(self, sample: dict):
        keys = self.apply_to if self.apply_to else sample.keys()
        for key in keys:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            if isinstance(item, Image):
                if key == 'label':
                    sample[key] = tr.Resize(self.size, interpolation=0)(item)
                else:
                    sample[key] = tr.Resize(self.size)(item)
            elif isinstance(item, (np.ndarray, np.generic)):
                c = item.shape[0]
                sample[key] = item.resize((c, *self.size))
            elif isinstance(item, str):
                pass
            else:
                raise NotImplementedError(type(item) + ' is not supported')
        return sample


class ToTensor(object):
    def __init__(self, apply_to=None):
        self.apply_to = apply_to

    def __call__(self, sample: dict):
        keys = self.apply_to if self.apply_to else sample.keys()
        for key in keys:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            if isinstance(item, Image):
                sample[key] = tr.ToTensor()(item)
            elif isinstance(item, (np.ndarray, np.generic)):
                sample[key] = torch.from_numpy(item)
            elif isinstance(item, str):
                pass
            else:
                raise NotImplementedError(type(item) + ' is not supported')
        return sample


class Squeeze(object):
    def __init__(self, apply_to: list):
        self.apply_to = apply_to

    def __call__(self, sample):
        for key in self.apply_to:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            assert isinstance(item, torch.Tensor) or isinstance(item, (np.ndarray, np.generic))
            sample[key] = item.squeeze() * 255
        return sample


class Normalize(object):
    def __init__(self, mean, std, apply_to: list):
        self.mean = mean
        self.std = std
        self.normalizer = tr.Normalize(mean=mean, std=std)
        self.apply_to = apply_to

    def __call__(self, sample):
        for key in self.apply_to:
            if key not in sample.keys():
                raise NotImplementedError
            item = sample[key]
            if isinstance(item, torch.Tensor):
                sample[key] = self.normalizer(item)
            else:
                raise NotImplementedError(type(item) + ' is not supported')
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p, apply_to=None):
        self.p = p
        self.apply_to = apply_to

    def __call__(self, sample):
        p = random.random()
        if p > self.p:
            keys = self.apply_to if self.apply_to else sample.keys()
            for key in keys:
                if key not in sample.keys():
                    raise NotImplementedError
                item = sample[key]
                if isinstance(item, Image):
                    sample[key] = tr.RandomHorizontalFlip(p=1)(item)
                elif isinstance(item, (np.ndarray, np.generic)):
                    sample[key] = item.fliplr()
                elif isinstance(item, str):
                    pass
                else:
                    raise NotImplementedError(type(item) + ' is not supported')
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p, apply_to=None):
        self.p = p
        self.apply_to = apply_to

    def __call__(self, sample):
        p = random.random()
        if p > self.p:
            keys = self.apply_to if self.apply_to else sample.keys()
            for key in keys:
                if key not in sample.keys():
                    raise NotImplementedError
                item = sample[key]
                if isinstance(item, Image):
                    sample[key] = tr.RandomVerticalFlip(p=1)(item)
                elif isinstance(item, (np.ndarray, np.generic)):
                    sample[key] = item.flipud()
                elif isinstance(item, str):
                    pass
                else:
                    raise NotImplementedError(type(item) + ' is not supported')
        return sample
