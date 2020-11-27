# camera-degradation
Repo of Camera Degradation task for Autopilot project

## Resnet34 and Resnet50

### Requirements
-----------
  - Python >= 3.6
  - Pytorch = 1.6.0+cu101


### Datasets
-----------
  - The CIFAR100 is already provided by torchvision
  - [The Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)

#### Download dataset
To download the `Intel Image Classification` dataset, follow these instructions:
  - Install kaggle
``` bash
$ pip install kaggle
```
  - `Create New API Token` on kaggle website
  - Copy the `kaggle.json` file into `.kaggle` folder in your computer
  - Download the dataset
``` bash
$ kaggle datasets download -d puneet6060/intel-image-classification
```


### Training
-----------
  - Clone project
  - In the resnet-pytorch directory, unzip dataset (if using Intel Image Classification for training):
``` bash
$ mkdir data
$ unzip intel-image-classification.zip -d data/
```
  - Train on CIFAR100
``` bash
$ python train.py -net=resnet34
```
or
``` bash
$ python train.py -net=resnet50
```
  - Train on Intel Image Classification
``` bash
$ python train_intel.py -net=resnet34
``` 
or 
``` bash
$ python train_intel.py -net=resnet50
```

### Testing
``` bash
$ python test.py -net=resnet34 -weights path_to_vgg16_weights_file
```
or
``` bash
$ python test.py -net=resnet50 -weights path_to_vgg16_weights_file
``` 
The test script for Intel dataset will be updated later.


### Implemented Network
  - [ResNet34 and ResNet50](https://arxiv.org/abs/1512.03385v1)

### Training details
  Followed the [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divided by 5 at 60th, 120th, 160th epochs. Train for 200 epochs, batch size 128 and weight decay 5e-4, Nesterov momentum of 0.9.
