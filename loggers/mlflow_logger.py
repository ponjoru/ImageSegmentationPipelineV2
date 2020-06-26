import mlflow
import os
import matplotlib.pyplot as plt
from loggers.logger_template import LoggerTemplate


class MLFlowLogger(LoggerTemplate):
    def __init__(self, settings, settings_to_log):
        super(MLFlowLogger, self).__init__()
        self.settings = settings
        self.settings_to_log = settings_to_log
        self.mlflow_settings = {
            'mlflow_s3_endpoint_url': None,
            'aws_access_key_id': None,
            'aws_secret_access_key': None,
            'mlflow_tracking_uri': None,
            'mlflow_experiment': "Test",
        }
        self.init_settings()
        self.log_experiment_config()

    def init_settings(self):
        """
        define MLflow settings
        :return:
        """
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.mlflow_settings['mlflow_s3_endpoint_url']
        os.environ['AWS_ACCESS_KEY_ID'] = self.mlflow_settings['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.mlflow_settings['aws_secret_access_key']
        mlflow.set_tracking_uri(self.mlflow_settings['mlflow_tracking_uri'])
        mlflow.set_experiment(self.mlflow_settings['mlflow_experiment'])

    def log_metrics(self, metrics_to_log: list, metrics_dict: dict, epoch: int):
        """
        Log metrics to MLflow
        :param metrics_to_log: list of metrics to log
        :param metrics_dict: dict of metrics and their values evaluated during the experiment
        :return:
        """
        try:
            for key in metrics_dict.keys():
                if key in metrics_to_log:
                    mlflow.log_metric(key.upper(), metrics_dict[key])
        except Exception as e:
            Warning('Failed to log metrics:', e)

    def log_metric(self, metric_tuple):
        assert isinstance(metric_tuple[0], str)
        try:
            mlflow.log_metric(metric_tuple[0], metric_tuple[1])
        except Exception as e:
            Warning('Failed to log %s:' % metric_tuple[0], e)

    def log_experiment_config(self):
        """
        Log basic information about the run to MLflow
        :param settings:
        :param settings_to_log:
        :param class_weight:
        :return:
        """
        try:
            for key in self.settings.keys():
                if key in self.settings_to_log:
                    mlflow.log_param(key.upper(), self.settings[key])
        except Exception as e:
            Warning('Failed to log metadata: ', e)

    def log_param(self, param_tuple):
        assert isinstance(param_tuple[0], str)
        try:
            mlflow.log_param(param_tuple[0], param_tuple[1])
        except Exception as e:
            Warning('Failed to log %s:' % param_tuple[0], e)

    def log_artifact(self, figure, epoch, name):
        assert isinstance(figure, plt.Figure)
        try:
            tmp_img_path = './' + name
            artifact_path = "%s/specials/" % epoch
            figure.savefig(tmp_img_path, bbox_inches='tight', dpi=figure.dpi)
            mlflow.log_artifact(tmp_img_path, artifact_path)
            os.remove(tmp_img_path)
        except Exception as e:
            print('Failed to log artifact', e)

    def log_checkpoint(self, state, key_metric, filename='checkpoint'):
        pass

    def close(self):
        pass