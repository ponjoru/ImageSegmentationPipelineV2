from abc import ABC, abstractmethod


class LoggerTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log_metrics(self, metrics_to_log: list, metrics_dict: dict, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def log_checkpoint(self, state, key_metric, filename='checkpoint'):
        raise NotImplementedError

    @abstractmethod
    def log_param(self, param_tuple):
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, figure, epoch, name):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass
