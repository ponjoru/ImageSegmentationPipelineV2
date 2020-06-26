from dataloaders.datasets import cityscapes
from torch.utils.data import DataLoader


def make_data_loader(settings, **kwargs):
    datasets = {
        'cityscapes': cityscapes.CityscapesSegmentation,
    }
    if settings['dataset'] not in datasets:
        raise NotImplementedError

    dataset_parser = datasets[settings['dataset']]
    train_set = dataset_parser(settings, split='train')
    val_set = dataset_parser(settings, split='val')
    test_set = dataset_parser(settings, split='test')
    num_class = train_set.num_classes

    batch_size = settings['batch_size']
    train_drop_last = len(train_set) % batch_size == 1

    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=train_drop_last, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=settings['val_batch_size'], shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    return loaders, num_class, train_set.plotter.plot_result