from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
from torch.optim.optimizer import Optimizer


class LRScheduler(object):

    def __init__(self, scheduler_type: str, optimizer: Optimizer, dataset_size: int, scheduler_params: dict = None):
        schedulers = {
            'multistep': MultiStepLR,
            'exponential': ExponentialLR,
            'reduceOnPlateau': ReduceLROnPlateau,
            'cosine': CosineAnnealingLR,
            'cyclic': CyclicLR,
        }

        default_params = {
            'multistep': {'milestones': [10, 20, 30, 40, 50], 'gamma': 0.1, 'last_epoch': -1},
            'exponential': {'gamma': 0.1, 'last_epoch': -1},
            'reduceOnPlateau': {'patience': 3, 'threshold': 0.001, 'mode': "max"},
            'cosine': {'T_max': 100, 'eta_min': 0, 'last_epoch': -1},
            'cyclic': {'base_lr': 1e-6, 'max_lr': 1e-3, 'step_size_up': 2000, 'step_size_down': None, 'mode': 'triangular', 'gamma': 0.6, 'scale_fn': None, 'scale_mode': 'cycle', 'cycle_momentum': True, 'base_momentum': 0.8, 'max_momentum': 0.9, 'last_epoch': -1},
        }

        if scheduler_type not in schedulers:
            raise NotImplementedError()

        if scheduler_params is not None:
            params = scheduler_params
        else:
            params = default_params[scheduler_type]

        self.scheduler_type = scheduler_type
        self.dataset_size = dataset_size
        self.per_batch_scheduler = scheduler_type in ['cyclic', 'cosine']
        self.scheduler = schedulers[scheduler_type](optimizer=optimizer, **params)
        self.epoch = 0

    def __call__(self, i, epoch, best_pred):
        if self.per_batch_scheduler:
            self.scheduler.step()
        elif epoch > self.epoch:
            self.epoch = epoch
            if self.scheduler_type == 'reduceOnPlateau':
                self.scheduler.step(best_pred)
            else:
                self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict=state_dict)
