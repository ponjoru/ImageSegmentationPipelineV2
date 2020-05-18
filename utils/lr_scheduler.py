from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
from torch.optim.optimizer import Optimizer

# TODO: add Cosine annealing


class LRScheduler(object):

    def __init__(self, scheduler_type: str, optimizer: Optimizer, batch_size: int, scheduler_params: dict = None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError
        if scheduler_params is not None:
            schedulers = {
                'multistep': MultiStepLR(optimizer, **scheduler_params),
                'exponential': ExponentialLR(optimizer, **scheduler_params),
                'reduceOnPlateau': ReduceLROnPlateau(optimizer, **scheduler_params),
                'cyclic': CyclicLR(optimizer, **scheduler_params),
            }
        else:
            schedulers = {
                'multistep': MultiStepLR(optimizer, [10, 20, 30, 40, 50], gamma=0.1, last_epoch=-1),
                'exponential': ExponentialLR(optimizer, gamma=0.1, last_epoch=-1),
                'reduceOnPlateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08),
                'cyclic': CyclicLR(optimizer, 1e-6, 1e-3, step_size_up=2000, step_size_down=None, mode='triangular', gamma=0.6, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1),
            }
        if scheduler_type not in schedulers:
            raise NotImplementedError()

        self.scheduler_type = scheduler_type
        self.batch_size = batch_size
        self.per_batch_scheduler = scheduler_type in ['cyclic']
        self.scheduler = schedulers[scheduler_type]
        self.epoch = 0

    def __call__(self, i, epoch, best_pred):
        if self.per_batch_scheduler:
            if not i % self.batch_size:
                self.scheduler.step()
        else:
            if epoch > self.epoch:
                if self.scheduler_type == 'reduceOnPlateau':
                    self.scheduler.step(best_pred)
                else:
                    self.scheduler.step()
