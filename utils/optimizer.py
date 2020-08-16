import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.sgd import SGD
from torch.optim.rmsprop import RMSprop


def define_optimizer(model: nn.Module, opt_name: str, opt_params: dict = None):
    if isinstance(model, nn.DataParallel):
        module = model.module
    else:
        module = model
    param_getter = getattr(module, "get_params", None)
    if callable(param_getter):
        param_list = module.get_params()
    else:
        param_list = module.params()
    if all([len(group['params']) == 0 for group in param_list]):
        raise Warning('No model parameters found')

    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'adadelta': Adadelta,
        'rmsprop': RMSprop
    }
    default_params = {
        'sgd': {"lr": 0.001, "momentum": 0, "dampening": 0, "weight_decay": 0, "nesterov": False},
        'adam': {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0, "amsgrad": False},
        'adadelta': {"lr": 1.0, "rho": 0.9, "eps": 1e-06, "weight_decay": 0},
        'rmsprop': {"lr": 0.01, "alpha": 0.99, "eps": 1e-08, "weight_decay": 0, "momentum": 0, "centered": False}
    }
    if opt_name not in optimizers:
        raise NotImplementedError(''.join(optimizers.keys()) + ': optimizers are available, given %s' % opt_name)

    if opt_params is None:
        opt_params = default_params[opt_name]

    return optimizers[opt_name](param_list, **opt_params)
