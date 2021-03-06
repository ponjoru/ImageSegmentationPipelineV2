import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders.make_data_loader import make_data_loader
from models.get_model import get_model
from settings import define_settings
from utils.optimizer import define_optimizer
from utils.lr_scheduler import LRScheduler
from utils.evaluator import Evaluator
from utils._utils import tensors_to_numpy, plot_confusion_matrix
from losses.custom_loss import CustomLoss
from loggers.main_logger import MainLogger
from dataloaders.custom_transforms import denormalize_image
from utils.mix_regularization import random_joint_mix, mix_criterion


class Trainer(object):
    def __init__(self, settings: dict, settings_to_log: list):
        self.settings = settings
        self.settings_to_log = settings_to_log

        self.threshold = self.settings['threshold']
        self.start_epoch = self.settings['start_epoch']
        self.dataset = self.settings['dataset']
        self.batch_size = self.settings['batch_size']
        self.workers = self.settings['workers']
        self.cuda = self.settings['cuda']
        self.fp16 = self.settings['fp16']
        self.epochs = self.settings['epochs']
        self.ignore_index = self.settings['ignore_index']
        self.loss_reduction = self.settings['loss_reduction']

        # -------------------- Define Data loader ------------------------------
        self.loaders, self.nclass, self.plotter = make_data_loader(settings)
        self.train_loader, self.val_loader, self.test_loader = [self.loaders[key] for key in ['train', 'val', 'test']]

        # -------------------- Define model ------------------------------------
        self.model = get_model(self.settings)

        # -------------------- Define optimizer and its options ----------------
        self.optimizer = define_optimizer(self.model, self.settings['optimizer'], self.settings['optimizer_params'])
        if self.settings['lr_scheduler']:
            self.lr_scheduler = LRScheduler(self.settings['lr_scheduler'], self.optimizer, self.batch_size)

        # -------------------- Define loss -------------------------------------
        input_size = (self.batch_size, self.nclass, *self.settings['target_size'])
        self.criterion = CustomLoss(input_size=input_size, ignore_index=self.ignore_index, reduction=self.loss_reduction)

        self.evaluator = Evaluator(metrics=self.settings['metrics'], num_class=self.nclass, threshold=self.settings['threshold'])

        self.logger = MainLogger(loggers=self.settings['loggers'], settings=settings, settings_to_log=settings_to_log)
        if self.settings['resume']:
            self.resume_checkpoint(self.settings['resume'])

        self.metric_to_watch = 0.0

    def activation(self, output):
        if self.nclass == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)
        return output

    def prepare_inputs(self, *inputs):
        if self.settings['cuda']:
            inputs = [i.cuda() for i in inputs]
        if self.settings['fp16']:
            inputs = [i.half() for i in inputs]
        return inputs

    def training(self, epoch: int):
        """
        Training loop for a certain epoch
        :param epoch: epoch id
        :return:
        """
        self.evaluator.reset()
        self.model.train()
        tbar = tqdm(self.train_loader, desc='train', file=sys.stdout)
        train_loss = 0.0
        output = {}
        for i, sample in enumerate(tbar):
            img, target = self.prepare_inputs(sample['image'], sample['label'])
            img, target, perm_target, gamma = random_joint_mix(img, target, self.settings['CutMix'], self.settings['MixUp'], p=self.settings['MixP'])

            self.optimizer.zero_grad()
            output['pred'], output['pred8'], output['pred16'] = self.model(img)

            if self.settings['MixUp'] or self.settings['CutMix']:
                loss = mix_criterion(self.criterion.train_loss, output, tgt_a=target, tgt_b=perm_target, gamma=gamma)
            else:
                loss = self.criterion.train_loss(**output, target=target)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()

            if self.settings['lr_scheduler']:
                self.lr_scheduler(i, epoch, self.metric_to_watch)

            out = self.activation(output['pred'])
            self.evaluator.add_batch(out, target)
            tbar.set_description('Train loss: %.4f, Epoch: %d' % (train_loss / float(i + 1), epoch))

            self.logger.log_metric(metric_tuple=('TRAIN_LOSS', (train_loss / float(i + 1))))
        _ = self.evaluator.eval_metrics(reduction=self.settings['evaluator_reduction'], show=True)

    def validation(self, epoch: int):
        """
        Validation loop for a certain epoch
        :param epoch: epoch id
        :return:
        """
        self.evaluator.reset()
        self.model.eval()
        if self.settings['validation_only']:
            loader = self.loaders[self.settings['validation_only']]
        else:
            loader = self.val_loader
        tbar = tqdm(loader, desc='valid', file=sys.stdout)
        test_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                img, target = self.prepare_inputs(sample['image'], sample['label'])

                output = self.model(img)

                loss = self.criterion.val_loss(pred=output, target=target)
                test_loss += loss.item()

                output = self.activation(output)
                self.evaluator.add_batch(output, target)
                tbar.set_description('Validation loss: %.3f, Epoch: %d' % (test_loss / (i + 1), epoch))

                if self.settings['log_artifacts']:
                    self.log_artifacts(epoch=epoch, sample=sample, output=output)

                self.logger.log_metric(metric_tuple=('VAL_LOSS', test_loss / (i + 1)))
        metrics_dict = self.evaluator.eval_metrics(reduction=self.settings['evaluator_reduction'], show=True)
        metrics_dict['val_loss'] = test_loss / (i + 1)
        self.metric_to_watch = metrics_dict[self.settings['metric_to_watch']].mean()
        if not self.settings['validation_only']:
            self.save_checkpoint(epoch=epoch, metrics_dict=metrics_dict)

    def save_checkpoint(self, epoch, metrics_dict):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.cuda else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics_dict,
            'scheduler': self.lr_scheduler.state_dict() if self.settings['lr_scheduler'] else None,
        }
        self.logger.log_metrics(self.settings['metrics'], metrics_dict, epoch=epoch)
        self.logger.log_checkpoint(state, key_metric=self.metric_to_watch, filename=self.settings['check_suffix'])

    def log_artifacts(self, sample, output, epoch):
        last_epoch = epoch == (self.settings['epochs'] - 1)
        if epoch % self.settings['log_dilate'] == 0 or last_epoch:
            sample['image'] = denormalize_image(sample['image'], **self.settings['normalize_params'])
            image, target, output = tensors_to_numpy(sample['image'], sample['label'], output)
            for ind, value in enumerate(sample['id']):
                if value in self.settings['inputs_to_watch']:
                    fig = self.plotter(image[ind], output[ind], target[ind],
                                       alpha=0.4, threshold=self.threshold, show=self.settings['show_results'])
                    self.logger.log_artifact(artifact=fig, epoch=epoch, name=value.replace('_leftImg8bit', ''))
                    plt.close()

    def resume_checkpoint(self, resume):
        if not os.path.isfile(resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
        checkpoint = torch.load(resume)
        self.start_epoch = checkpoint['epoch']
        if self.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if not self.settings['fine_tuning']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['scheduler']:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.metric_to_watch = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch: {}, best_metric: {:.4f})"
              .format(resume, checkpoint['epoch'], self.metric_to_watch))

    def close(self):
        fig = plot_confusion_matrix(self.evaluator.confusion_matrix, normalize=True, title=None, cmap=plt.cm.Blues, show=False)
        self.logger.log_artifact(fig, epoch=-1, name='confusion_matrix.png')
        self.logger.close()


def main():
    # define run settings
    settings, settings_to_log = define_settings()
    trainer = Trainer(settings, settings_to_log)

    if settings['validation_only']:
        trainer.validation(0)
        return

    print('Starting Epoch: ', trainer.start_epoch, '; ', 'Total Epochs: ', trainer.epochs)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        print('\n ------------------------------------ EPOCH %d ------------------------------------ ' % epoch)
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.close()


if __name__ == "__main__":
    main()
