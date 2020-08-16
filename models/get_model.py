import torch
import torch.nn as nn
from models.bisenet_v1 import BiSeNetV1
from models.deeplabv3plus import DeepLabv3_plus


def get_model(settings):
    name = settings['model_name']
    models = {
        'BiSeNetV1': BiSeNetV1,
        'DeepLabV3+': DeepLabv3_plus,
    }
    if name not in models.keys():
        raise NotImplementedError

    model = models[name](**settings['model_kwargs'])

    if settings['freeze_backbone']:
        model.freeze_backbone()

    if settings['fp16']:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if settings['cuda']:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    return model

