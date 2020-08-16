from loggers.logger_template import LoggerTemplate
import os
import glob
import shutil
import torch
from utils.utils_test import metrics2str


class LocalLogger(LoggerTemplate):
    def __init__(self, settings, settings_to_log, max_id: int = 20, img_save_path=None, save_last_img_only=True):
        super(LocalLogger, self).__init__()
        self.settings = settings
        self.settings_to_log = settings_to_log
        self.img_save_path = img_save_path
        self.save_last_img_only = save_last_img_only

        self.dataset = settings['dataset']
        self.filename = settings['model_name'].lower().rstrip()

        self.directory = os.path.join('run', self.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1][0]) + 1 if self.runs else 0
        run_id = min(run_id, max_id)

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.log_experiment_config()

        self.best_metric = 0.0
        self.best_metrics = {}

    def log_metrics(self, metrics_to_log: list, metrics_dict: dict, epoch: int):
        pass

    def log_metric(self, metric_tuple):
        pass

    def log_param(self, param_tuple):
        pass

    def log_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        for key in self.settings.keys():
            log_file.write(key + ':' + str(self.settings[key]) + '\n')
        log_file.close()

    def log_checkpoint(self, state, key_metric, filename='checkpoint'):
        """Saves checkpoint to disk"""
        is_best = False
        if key_metric > self.best_metric:
            self.best_metric = key_metric
            self.best_metrics = state['metrics']
            is_best = True

        checkpoint_name = self.filename + '_' + filename + '.pth.tar'
        filename = os.path.join(self.experiment_dir, checkpoint_name)

        state['best_pred'] = self.best_metric
        torch.save(state, filename)

        with open(os.path.join(self.experiment_dir, 'history.txt'), 'a') as f:
            f.write('{:3s}: {:.5f}\n'.format(str(state['epoch']), key_metric))
        f.close()

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('best prediction: %.4f\n' % best_pred)
                f.write(metrics2str(state['metrics']))
            f.close()
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            print('\033[94m'+'Best checkpoint was saved'+'\033[0m')

        print('Checkpoint was saved')

    def log_artifact(self, artifact, epoch, name):
        if self.img_save_path is None:
            dst_path = os.path.join(self.experiment_dir, name)
        else:
            dst_path = os.path.join(self.img_save_path, name)

        if not self.save_last_img_only:
            dst_path = os.path.join(dst_path, str(epoch))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

        artifact.savefig(dst_path, dpi=artifact.dpi)

    def close(self):
        self.rename_best_checkpoint(filename='best')

    def rename_best_checkpoint(self, filename='best_checkpoint'):
        metric = '{:1.4f}'.format(self.best_metric).replace('.', '_')
        src = os.path.join(self.experiment_dir, 'model_best.pth.tar')
        dst = os.path.join(self.experiment_dir, self.settings['dataset'] + '_' + self.filename + '@' + metric + '_' + filename + '.pth.tar')
        os.rename(src, dst)

