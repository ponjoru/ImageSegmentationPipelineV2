import torch


def define_settings():
    settings = {
        # --------------- Global settings ----------------
        'validation_only':      False,  # OneOf('val', 'test', 'train', False) - runs only validation
        'threshold':            0.5,
        'start_epoch':          0,
        'epochs':               40,
        'seed':                 1,
        'cuda':                 True,
        'loss_reduction':       'mean',
        'evaluator_reduction':  'none',
        'optimizer':            'adam',
        'optimizer_params':     None,
        'lr_scheduler':         None,  # 'multistep', 'exponential', 'reduceOnPlateau', 'cyclic'
        'scheduler_params':     None,
        'metrics':              ['iou', 'dice'],
        'metric_to_watch':      'iou',
        'log_dilate':           1,
        'inputs_to_watch':      ['frankfurt_000001_013016_leftImg8bit.png', 'frankfurt_000001_029086_leftImg8bit.png',
                                 'frankfurt_000001_042733_leftImg8bit.png', 'frankfurt_000001_067735_leftImg8bit.png',
                                 'lindau_000030_000019_leftImg8bit.png', 'lindau_000050_000019_leftImg8bit.png',
                                 'frankfurt_000000_011074_leftImg8bit.png', 'frankfurt_000001_055172_leftImg8bit.png',
                                 'frankfurt_000001_065850_leftImg8bit.png', 'frankfurt_000001_060422_leftImg8bit.png',
                                 'frankfurt_000001_068208_leftImg8bit.png'],
        'show_results':         False,
        'log_artifacts':        True,
        'fine_tuning':          False,      # if True optimizer and scheduler are reinitialized, otherwise their states are loaded from the checkpoint
        'resume':               '/home/user/rzd/ImageV2/run/cityscapes/experiment_0@0_4609_10epochs/bisenetv1_checkpoint.pth.tar',       # path to the checkpoint
        'fp16':                 False,
        'loggers':              ['local'],      # local, mlflow
        'MixUp':                False,
        'CutMix':               False,
        'MixP':                 0.5,  # if MixUp and CutMix = True, with probability p chooses MixUp, and 1-p - CutMix for a batch

        # ----------------- ANN settings -----------------
        'model_name':           'BiSeNetV1',
        'model_kwargs':         {'n_classes': 19},
        'check_suffix':         'test_run',
        'comments':             'first run',
        'freeze_backbone':      False,

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
        'workers':              10,
    }
    # list of common settings that should be logged
    settings_to_log = settings.keys()

    # check if cuda is available
    if settings['cuda'] and not torch.cuda.is_available():
        print('Failed to use cuda')
        settings['cuda'] = False

    print("Running training with arguments:")
    print(settings)
    torch.manual_seed(settings["seed"])

    return settings, settings_to_log
