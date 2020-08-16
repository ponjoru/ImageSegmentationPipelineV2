import torch.nn as nn
from abc import ABC, abstractmethod


class ModelTemplate(nn.Module, ABC):
    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    def freeze_backbone(self):
        if hasattr(self, 'backbone') and isinstance(self.backbone, nn.Module):
            for layer in self.backbone.children():
                layer.requires_grad = False
        else:
            print('\033[93mWarning: Failed to freeze backbone.\033[0m')

    def freeze_named_modules(self, named_modules):
        for name in named_modules:
            module = getattr(self, name, None)
            if module and isinstance(module, nn.Module):
                for layer in module.children():
                    layer.requires_grad = False
            else:
                print('\033[93mWarning: Failed to freeze ' + name + '.\033[0m')