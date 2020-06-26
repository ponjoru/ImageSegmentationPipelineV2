from loggers.logger_template import LoggerTemplate
from loggers.mlflow_logger import MLFlowLogger
from loggers.local_logger import LocalLogger


class MainLogger(LoggerTemplate):
    def __init__(self, loggers, settings, settings_to_log):
        super(MainLogger, self).__init__()

        self.loggers_instances = {
            'mlflow': MLFlowLogger,
            'local': LocalLogger,
        }

        self.logger_params = {
            'mlflow': {"settings": settings, "settings_to_log": settings_to_log},
            'local': {"settings": settings, "settings_to_log": settings_to_log, "img_save_path": './results'}
        }
        if any([logger not in self.loggers_instances.keys() for logger in loggers]):
            raise NotImplementedError

        self.loggers = {}
        for logger_name in loggers:
            self.loggers[logger_name] = self.loggers_instances[logger_name](**self.logger_params[logger_name])

    def log_artifact(self, artifact, epoch, name):
        for logger in self.loggers.values():
            logger.log_artifact(artifact, epoch, name)

    def log_metrics(self, metrics_to_log: list, metrics_dict: dict, epoch: int):
        for logger in self.loggers.values():
            logger.log_metrics(metrics_to_log, metrics_dict, epoch)

    def log_metric(self, metric_tuple):
        for logger in self.loggers.values():
            logger.log_metric(metric_tuple)

    def log_checkpoint(self, state, key_metric, filename='checkpoint'):
        for logger in self.loggers.values():
            logger.log_checkpoint(state, key_metric, filename='checkpoint')

    def log_param(self, param_tuple):
        for logger in self.loggers.values():
            logger.log_param(param_tuple)

    def close(self):
        for logger in self.loggers.values():
            logger.close()
