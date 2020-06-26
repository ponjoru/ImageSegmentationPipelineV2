import sys
import os
import torch
from tqdm import tqdm
from dataloaders.make_data_loader import make_data_loader
from models.deeplabv3plus import DeepLabv3_plus
from settings import define_settings
from loggers.checkpoint_saver import CheckpointSaver
from utils.optimizer import define_optimizer
from utils.lr_scheduler import LRScheduler
from utils.evaluator import Evaluator
from utils.utils_test import tensors_to_numpy
from losses.custom_loss import CustomLoss


class Trainer(object):
    def __init__(self, settings: dict):
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
        self.model = DeepLabv3_plus(nInputChannels=3, n_classes=self.nclass, pretrained=True)

        # -------------------- Moving to GPU -----------------------------------
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        # -------------------- Define optimizer and its options ----------------
        self.optimizer = define_optimizer(self.model, self.settings['optimizer'], self.settings['optimizer_params'])
        if self.settings['lr_scheduler']:
            self.lr_scheduler = LRScheduler(self.settings['lr_scheduler'], self.optimizer, self.batch_size)

        # -------------------- Define loss -------------------------------------
        self.criterion = CustomLoss(ignore_index=self.ignore_index, reduction=self.loss_reduction)

        self.evaluator = Evaluator(metrics=self.settings['metrics'], num_class=self.nclass, threshold=self.settings['threshold'], cuda=self.cuda)
        self.saver = CheckpointSaver(settings=settings, keywords_to_save=settings, max_id=10)

        if self.settings['resume']:
            self.resume_checkpoint(self.settings['resume'])

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

            if i == 10:
                break
        metrics_dict = self.evaluator.eval_metrics(reduction=self.settings['evaluator_reduction'], show=True)

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

        for i, sample in enumerate(tbar):
            img, target = sample['image'], sample['label']
            if self.cuda:
                img, target = img.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(img)

            loss = self.criterion.val_loss(input=output, target=target, epoch=epoch)
            test_loss += loss.item()

            output = self.activation(output)
            self.evaluator.add_batch(output, target)
            tbar.set_description('Validation loss: %.3f, Epoch: %d' % (test_loss / (i + 1), epoch))

            self.plot_and_log_result(epoch=epoch, sample=sample, output=output)
        metrics_dict = self.evaluator.eval_metrics(reduction=self.settings['evaluator_reduction'], show=True)
        self.save_checkpoint(epoch=epoch, key_metric=metrics_dict[self.settings['metric_to_watch']].mean())

    def activation(self, output):
        if self.nclass == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)
        return output

    def save_checkpoint(self, epoch, key_metric):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.cuda else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': {None},
        }
        self.saver.save_checkpoint(state, key_metric=key_metric, filename=self.settings['check_suffix'])

    def plot_and_log_result(self, epoch, sample, output):
        if epoch % self.settings['log_dilate'] == 0 or epoch == (self.settings['epochs'] - 1):
            image, target, output = tensors_to_numpy(sample['image'], sample['label'], output, cuda=self.cuda)
            for ind, value in enumerate(sample['id']):
                if value in self.settings['inputs_to_watch']:
                    fig = self.plotter(image[ind], output[ind], target[ind],
                                       alpha=0.4, threshold=self.threshold, show=self.settings['show_results'],
                                       save=self.settings['save_pict'], img_name=value)

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
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))


def main():
    # define run settings
    settings, settings_to_log = define_settings()
    trainer = Trainer(settings)

    print('Starting Epoch: ', trainer.start_epoch, '; ', 'Total Epochs: ', trainer.epochs)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        print('\n ------------------------------------ EPOCH %d ------------------------------------ ' % epoch)
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.saver.rename_best_checkpoint(filename='best')


if __name__ == "__main__":
    main()
