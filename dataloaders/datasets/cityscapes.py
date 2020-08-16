from dataloaders.datasets.dataset_template import ISDatasetTemplate
import os
from PIL import Image
from mypath import Path
from torchvision import transforms
import dataloaders.custom_transforms as ctr
from utils._utils import recursive_glob, one_hot_encode, alb_transform_wrapper
from dataloaders.label_creator import label_creator
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CityscapesSegmentation(ISDatasetTemplate):
    """
    Cityscapes (https://www.cityscapes-dataset.com/) data preparation class of 35 classes. The dataloader supports
    multiclass and binary segmentation, and flexible choice of classes to use, by defining valid_classes settings or
    changing in the code itself. In binary segmentation mode all the valid classes are merged into one: 'foreground',
    and output mask is binary matrix (W,H). In multicalss: classes that are outside of valid classes list are
    transferred to class 'background' with ignore index, and output mask is a matrix (W,H), with each element
    el: 0 <= el <= C-1, where C - the number of valid classes. Classes with equal train_id are treated as one

    Attention: defining valid_classes overrides the label table, initialized in the constructor. To use only the table:
    define valid_classes as None in the settings.

    :param settings: dict, to continue work with the dataloader following settings should be specified:
                     valid_classes, segmentation_mode, ignore_index, crop_size. For the detailed explanation please
                     refer to the settings.py
    :param root:  path to the dataset, default: path defined in the mypath.py
    :param split: one of ['train', 'val', 'test'], default: 'train'
    :returns sample['image']: (3,W,H)
             sample['label']: (W,H), with each element el: 0 <= el <= C-1, where C - the number of classes
             sample['id']: metadata - image name
    """

    NUM_CLASSES = 35

    def __init__(self, settings: dict, root: str = Path.db_root_dir('cityscapes'), split: str = "train"):
        self.valid_classes = settings['valid_classes']
        binary = settings['segmentation_mode'] == 'binary'
        self.ignore = settings['ignore_index']

        # Init namedtuple labels creator
        Label = label_creator(self.valid_classes, binary, ignore_index=self.ignore, binary_color=settings['binary_color'])
        #         name                      id      train_id    color
        labels = [
            Label('unlabeled',               0,  self.ignore,   (  0,   0,   0)),
            Label('ego vehicle',             1,  self.ignore,   (  0,   0,   0)),
            Label('rectification border',    2,  self.ignore,   (  0,   0,   0)),
            Label('out of roi',              3,  self.ignore,   (  0,   0,   0)),
            Label('static',                  4,  self.ignore,   (  0,   0,   0)),
            Label('dynamic',                 5,  self.ignore,   (111,  74,   0)),
            Label('ground',                  6,  self.ignore,   ( 81,   0,  81)),
            Label('road',                    7,            0,   (128,  64, 128)),
            Label('sidewalk',                8,            1,   (244,  35, 232)),
            Label('parking',                 9,  self.ignore,   (250, 170, 160)),
            Label('rail track',             10,  self.ignore,   (230, 150, 140)),
            Label('building',               11,            2,   ( 70,  70,  70)),
            Label('wall',                   12,            3,   (102, 102, 156)),
            Label('fence',                  13,            4,   (190, 153, 153)),
            Label('guard rail',             14,  self.ignore,   (180, 165, 180)),
            Label('bridge',                 15,  self.ignore,   (150, 100, 100)),
            Label('tunnel',                 16,  self.ignore,   (150, 120,  90)),
            Label('pole',                   17,            5,   (153, 153, 153)),
            Label('polegroup',              18,  self.ignore,   (153, 153, 153)),
            Label('traffic light',          19,            6,   (250, 170,  30)),
            Label('traffic sign',           20,            7,   (220, 220,   0)),
            Label('vegetation',             21,            8,   (107, 142,  35)),
            Label('terrain',                22,            9,   (152, 251, 152)),
            Label('sky',                    23,           10,   ( 70, 130, 180)),
            Label('person',                 24,           11,   (220,  20,  60)),
            Label('rider',                  25,           12,   (255,   0,   0)),
            Label('car',                    26,           13,   (  0,   0, 142)),
            Label('truck',                  27,           14,   (  0,   0,  70)),
            Label('bus',                    28,           15,   (  0,  60, 100)),
            Label('caravan',                29,  self.ignore,   (  0,   0,  90)),
            Label('trailer',                30,  self.ignore,   (  0,   0, 110)),
            Label('train',                  31,           16,   (  0,  80, 100)),
            Label('motorcycle',             32,           17,   (  0,   0, 230)),
            Label('bicycle',                33,           18,   (119,  11,  32)),
            Label('license plate',          -1,  self.ignore,   (  0,   0, 142)),
        ]
        super(CityscapesSegmentation, self).__init__(labels, binary, threshold=0.5)

        self.root = root
        self.split = split
        self.settings = settings
        self.crop_size = settings['target_size']

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files = dict()
        self.files[split] = recursive_glob(rootdir=self.annotations_base, suffix='_labelIds.png')

        self.transformations = {
            'train': self.transform_tr,  # self.alb_transform_tr,
            'val': self.transform_val,
            'test': self.transform_ts,
        }

        if self.split not in self.transformations:
            raise ValueError
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.annotations_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        lbl_path = self.files[self.split][index].rstrip()
        img_path = lbl_path.replace('gtFine_labelIds', 'leftImg8bit')
        img_path = img_path.replace('gtFine', 'leftImg8bit')

        _img = Image.open(img_path).convert('RGB')
        _target = Image.open(lbl_path).convert('L')
        _target = self.map_target(_target)
        transform = self.transformations[self.split]
        sample = {'image': _img,
                  'label': _target,
                  'id': os.path.basename(img_path)}

        return transform(sample)

    def alb_transform_tr(self, sample: dict):
        alb_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.3),
            A.Resize(*self.settings['target_size']),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, p=0.3),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.Normalize(**self.settings['normalize_params']),
            ToTensorV2(),
        ], additional_targets={'label': 'mask'})
        return alb_transform_wrapper(alb_transforms, sample)

    def transform_tr(self, sample: dict):
        sample_transforms = transforms.Compose([
            ctr.RandomCrop(size=self.settings['rnd_crop_size']),
            ctr.RandomHorizontalFlip(p=0.5),
            ctr.ToTensor(),
            ctr.Normalize(**self.settings['normalize_params'], apply_to=['image']),
            ctr.Squeeze(apply_to=['label']),
        ])
        return sample_transforms(sample)

    def transform_val(self, sample: dict):
        sample_transforms = transforms.Compose([
            ctr.ToTensor(),
            ctr.Normalize(**self.settings['normalize_params'], apply_to=['image']),
            ctr.Squeeze(apply_to=['label']),
        ])
        return sample_transforms(sample)

    def transform_ts(self, sample: dict):
        sample_transforms = transforms.Compose([
            ctr.ToTensor(),
            ctr.Normalize(**self.settings['normalize_params'], apply_to=['image']),
            ctr.Squeeze(apply_to=['label']),
        ])
        return sample_transforms(sample)


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from utils._utils import create_canvas
    from dataloaders.custom_transforms import denormalize_image

    settings = {
        'base_size': (1024, 2048),
        'target_size': (1024, 2048),
        'rnd_crop_size': (768, 768),
        'normalize_params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'valid_classes': None,  # [7, 24, 26], [*range(15), *range(17, 25)],
        'segmentation_mode': 'multiclass',
        'binary_color': (255, 255, 0),
        'ignore_index': 255,
    }

    cityscapes_train = CityscapesSegmentation(settings, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        sample["image"] = denormalize_image(sample["image"], **settings['normalize_params'])
        for jj in range(sample["image"].size()[0]):
            img = sample["image"].numpy()
            gt = sample['label'].numpy()
            title = sample['id']
            target = one_hot_encode(gt, num_classes=cityscapes_train.num_classes)
            fig, ((ax1, ax2), (ax3, ax4)) = create_canvas(2,2, 2,2, (768, 768))
            cityscapes_train.plotter.plot_inference(ax1, img[jj], target[jj], alpha=1)
            ax1.set_title('one-hot encoded mask plot')
            cityscapes_train.plotter.plot_ground_truth(ax2, img[jj], gt[jj], alpha=1)
            ax2.set_title('default mask plot')
            cityscapes_train.plotter.plot_ground_truth(ax3, img[jj], gt[jj], alpha=0.4)
            ax3.set_title('one-hot encoded mask blended with input image with 0.4 coeff')
            ax4.imshow(np.transpose(img[jj], (1, 2, 0)))
            ax4.set_title('input image')
            plt.show()

        if ii == 1:
            print("image shape: ", img.shape)
            print("mask shape: ", gt.shape)
            break

    plt.show(block=True)

