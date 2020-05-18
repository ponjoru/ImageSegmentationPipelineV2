import os
import shutil
import torch
import glob


class CheckpointSaver(object):

    def __init__(self, settings: dict, keywords_to_save: dict, max_id: int = 20):
        self.dataset = settings['dataset']
        self.settings = settings
        self.keywords = keywords_to_save
        self.filename = settings['model_name'].lower().rstrip()

        self.directory = os.path.join('run', self.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        run_id = min(run_id, max_id)

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_experiment_config()

        self.best_metric = 0.0

    def save_checkpoint(self, state, key_metric, filename='checkpoint'):
        """Saves checkpoint to disk"""
        is_best = False
        if key_metric > self.best_metric:
            self.best_metric = key_metric
            is_best = True

        checkpoint_name = self.filename + '_' + filename + '.pth.tar'
        filename = os.path.join(self.experiment_dir, checkpoint_name)

        state['best_pred'] = self.best_metric
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred) + '\n')
                f.write(str(state['metrics']))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            print('Best checkpoint was saved')

        print('Checkpoint was saved')

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        for key, val in self.keywords.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def rename_best_checkpoint(self, filename='best_checkpoint'):
        metric = '{:1.4f}'.format(self.best_metric).replace('.', '_')
        src = os.path.join(self.experiment_dir, 'model_best.pth.tar')
        dst = os.path.join(self.experiment_dir, self.filename + '_' + metric + '_' + filename + '.pth.tar')
        os.rename(src, dst)