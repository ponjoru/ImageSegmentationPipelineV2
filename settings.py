import torch


def define_settings():
    settings = {
        # --------------- Global settings ----------------
        'threshold':            0.5,
        'start_epoch':          0,
        'epochs':               2,
        'seed':                 1,
        'cuda':                 True,
        'loss_reduction':       'mean',
        'evaluator_reduction':  'none',
        'optimizer':            'adam',
        'optimizer_params':     None,
        'lr_scheduler':         None,  # 'multistep', 'exponential', 'reduceOnPlateau', 'cyclic'
        'scheduler_params':     None,
        'metrics':              ['hard_iou', 'hard_dice', 'soft_iou', 'soft_dice'],
        'metric_to_watch':      'hard_iou',
        'log_dilate':           1,
        'inputs_to_watch':      ['frankfurt_000001_013016_leftImg8bit.png', 'frankfurt_000001_029086_leftImg8bit.png',
                                 'frankfurt_000001_042733_leftImg8bit.png', 'frankfurt_000001_067735_leftImg8bit.png',
                                 'lindau_000030_000019_leftImg8bit.png', 'lindau_000050_000019_leftImg8bit.png',
                                 'frankfurt_000000_011074_leftImg8bit.png', 'frankfurt_000001_055172_leftImg8bit.png',
                                 'frankfurt_000001_065850_leftImg8bit.png', 'frankfurt_000001_060422_leftImg8bit.png',
                                 'frankfurt_000001_068208_leftImg8bit.png'],
        'show_results':         False,
        'save_pict':            True,
        'fine_tuning':          False,
        'resume':               None,

        # ----------------- ANN settings -----------------
        'model_name':           'DeepLabV3+',
        'backbone_name':        'resnet18',
        'check_suffix':         'test_run',

        # --------------- Dataset settings ---------------
        'batch_size':           6,
        'val_batch_size':       6,
        'dataset':              'cityscapes',
        'base_size':            (1024, 2048),
        'target_size':          (768, 768),
        'crop_params':          {'left': 0, 'right': 1, 'upper': 0, 'lower': 1},
        'rnd_crop_size':        (768, 768),
        'normalize_params':     {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'valid_classes':        None,
        'segmentation_mode':    'multiclass',
        'binary_color':         (255, 255, 0),
        'ignore_index':         255,
        'workers': 0,
    }
    # list of common settings that should be logged to the local files and mlflow
    settings_to_log = ['model_name', 'batch_size', 'epochs', 'dataset', 'base_size', 'crop_size', 'loss_type',
                       'optimizer', 'lr_scheduler', 'use_balanced_weights', 'metric_to_watch', 'valid_classes',
                       'segmentation_mode', 'backbone_name']

    # check if cuda is available
    if settings['cuda'] and not torch.cuda.is_available():
        print('Failed to use cuda')
        settings['cuda'] = False

    print("Running training with arguments:")
    print(settings)
    torch.manual_seed(settings["seed"])

    return settings, settings_to_log
