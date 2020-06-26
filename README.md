<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#pytorch-image-segmentation-pipeline)
* [Getting Started](#getting-started)
  * [Start training](#start-training)
  * [Add custom dataset](#add-custom-dataset)
  * [Add custom loss](#add-custom-loss)
  * [Add custom logger](#add-custom-logger)
  * [Add custom metric](#add-custom-metric)
* [Project structure](#project-structure)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## Pytorch Image Segmentation Pipeline
Current project contains an extendable end-to-end pipeline for Computer Vision Semantic Segmentation in 
PyTorch. \
The main features included:
1. Multiclass and binary segmentation pipeline
2. Different losses included s.t.: SoftJaccardLoss(Intersection over union loss), SoftDiceLoss (F1 loss), 
Lovasz loss, Online Hard Example Mining CrossEntropy Loss, Focal Loss.
4. Easy to use API for extending the pipeline by adding custom models, modules, backbones, datasets and loggers
5. Evaluator with supported metrics: IoU, Dice, Frequently weighted IoU, Accuracy, Precision, Recall (mean and per-class)
5. Saving checkpoints and artifacts with multiple Loggers: Local, MlFlow

TODO:
1. Add Label Smoothing
2. Add class weights calculation over a dataset, extend and all losses with weights
3. Add MixUp support
4. Add model zoo API
5. Add fp16 training mode
6. Test MlFlow logger, add Tensorboard
7. Replace custom transforms with albumentations lib


## Getting started
* Clone the repo 
```sh
git clone https://github.com/ponjoru/ImageSegmentationPipelineV2.git 
```
* Install the packages mentioned in the requirements.txt:
```text
efficientnet-pytorch==0.5.1
entrypoints==0.3
flow-vis==0.1
matplotlib==3.1.3
mlflow==1.6.0
numpy==1.17.4
opencv-contrib-python==4.2.0.34
opencv-python==4.2.0.32
Pillow==7.0.0
scikit-learn==0.22.1
scipy==1.4.1
torch==1.4.0
torchvision==0.5.0
tqdm==4.42.1
```
### Start training
1. Init the training settings in `settings.py`
2. Add your model to `train.py`
3. Run `train.py`

### Add custom dataset
1. Download a dataset.
2. Split the dataset into test, train and valid folders
3. Create dataloader and inherit the abstract class `dataloaders\datasets\dataset_template`.
    ```text
    Functions to overload:
        def __len__(self)
        def __getitem__(self, index)
        def transform_tr(self, sample)
        def transform_val(self, sample)
        def transform_ts(self, sample)
    ```
4. add path to your dataset at `mypath.py`
5. add the custom dataloader instance to `dataloaders\make_data_loader.py`

Refer to the `dataloaders\datasets\cityscapes.py` as an example.

### Add custom loss
To change custom loss modify `class CustomLoss` defined in `train.py`

### Add custom logger
1. Create logger and inherit the abstract class `loggers\logger_template.py`.
2. Add the logger instance to `loggers\main_logger.py`

### Add custom metric
To add custom metric override `class Evaluator` defined in `utils\evaluator.py` or extend it with needed metrics

## Project Structure
    .
    ├── dataloaders
    │   ├── datasets                    # dataloaders for datasets
    │       ├── cityscapes.py           # cityscapes dataset dataloader
    │       └── dataset_template.py     # template for a custom dataloader
    │   ├── custom_transforms.py        # custom augmentation transforms
    │   └── make_dataloader.py          # dataloader factory
    ├── loggers                         # loggers are stored there
    │   ├── main_logger.py              # main logger manager
    │   ├── logger_template.py          # template for a custom logger
    │   └── ...                         # various loggers
    ├── losses                          # custom loss functions
    │   ├── custom_loss.py              # main custom loss to use in training
    │   └── ...                         # custom losses
    ├── models                          # models and modules
    │   ├── backbones                   # feature extractors
    │   ├── modules                     # useful modules
    │   └── ...                         # deep learning models
    ├── pretrained                      # important weights stored there   
    ├── results                         # validation results stored here
    ├── run                             # history of experiments with saved models and their parameters. Split by datasets and backbones
    ├── utils                           # utilities
    ├── mypath.py                       # define available datasets
    ├── settings.py                     # training settings
    ├── train.py                        # train script
    └── README.md

## Contact
tg: @ponjoru, e-mail: ig.popov1997@gmail.com

Project Link: [https://github.com/ponjoru/ImageSegmentationPipelineV2.git](https://github.com/ponjoru/ImageSegmentationPipelineV2.git)