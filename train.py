import sys
import os
import torch
from tqdm import tqdm
from dataloaders.make_data_loader import make_data_loader
from models.deeplabv3plus import DeepLabv3_plus
from settings import define_settings
from loggers.main_logger import MainLogger
from utils.optimizer import define_optimizer
from utils.lr_scheduler import LRScheduler
from utils.evaluator import Evaluator
from utils.utils_test import tensors_to_numpy
from losses.custom_loss import CustomLoss
from dataloaders.custom_transforms import denormalize_image


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
        self.epochs = self.settings['epochs']
        self.ignore_index = self.settings['ignore_index']
        self.loss_reduction = self.settings['loss_reduction']

        # -------------------- Define Data loader ------------------------------
        loaders, self.nclass, self.plotter = make_data_loader(settings)
        self.train_loader, self.val_loader, self.test_loader = [loaders[key] for key in ['train', 'val', 'test']]

        # -------------------- Define model ------------------------------------
        self.model = get_model(self.settings)

        # -------------------- Define optimizer and its options ----------------
        self.optimizer = define_optimizer(self.model, self.settings['optimizer'], self.settings['optimizer_params'])
        if self.settings['lr_scheduler']:
            self.lr_scheduler = LRScheduler(self.settings['lr_scheduler'], self.optimizer, self.batch_size)

        # -------------------- Define loss -------------------------------------
        input_size = (self.batch_size, self.nclass, *self.settings['target_size'])
        self.criterion = CustomLoss(input_size=input_size, ignore_index=self.ignore_index, reduction=self.loss_reduction, delay=100)

        self.evaluator = Evaluator(metrics=self.settings['metrics'], num_class=self.nclass, threshold=self.settings['threshold'])

        self.logger = MainLogger(loggers=self.settings['loggers'], settings=settings, settings_to_log=settings_to_log)
        if self.settings['resume']:
            self.resume_checkpoint(self.settings['resume'])

    def activation(self, output):
        if self.nclass == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)
        return output

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

        for i, sample in enumerate(tbar):
            img, target = sample['image'], sample['label']
            if self.cuda:
                img, target = img.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(img)

            loss = self.criterion.train_loss(input=output, target=target, epoch=epoch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            output = self.activation(output)
            self.evaluator.add_batch(output, target)
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
        tbar = tqdm(self.val_loader, desc='valid', file=sys.stdout)
        test_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(img)

                loss = self.criterion.val_loss(input=output, target=target, epoch=epoch)
                test_loss += loss.item()

                output = self.activation(output)
                self.evaluator.add_batch(output, target)
                tbar.set_description('Validation loss: %.3f, Epoch: %d' % (test_loss / (i + 1), epoch))

                if self.settings['log_artifacts']:
                    self.log_artifacts(epoch=epoch, sample=sample, output=output)

                self.logger.log_metric(metric_tuple=('VAL_LOSS', test_loss / (i + 1)))
        metrics_dict = self.evaluator.eval_metrics(reduction=self.settings['evaluator_reduction'], show=True)
        self.save_checkpoint(epoch=epoch, metrics_dict=metrics_dict)

    def save_checkpoint(self, epoch, metrics_dict):
        key_metric = metrics_dict[self.settings['metric_to_watch']].mean()
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.cuda else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics_dict,
        }
        self.logger.log_metrics(self.settings['metrics'], metrics_dict, epoch=epoch)
        self.logger.log_checkpoint(state, key_metric=key_metric, filename=self.settings['check_suffix'])

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
        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch: {}, best_metric: {.4f})"
              .format(resume, checkpoint['epoch'], self.best_pred))


def main():
    # define run settings
    settings, settings_to_log = define_settings()
    trainer = Trainer(settings, settings_to_log)

    print('Starting Epoch: ', trainer.start_epoch, '; ', 'Total Epochs: ', trainer.epochs)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        print('\n ------------------------------------ EPOCH %d ------------------------------------ ' % epoch)
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.logger.close()


if __name__ == "__main__":
    main()
