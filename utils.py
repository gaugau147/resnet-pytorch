import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    '''
    return the required network
    '''

    if args.net == 'resnet34':
        from model import resnet34
        net = resnet34()

    elif args.net == 'resnet50':
        from model import resnet50
        net = resnet50()

    else:
        print(args.net + ' is not supported, please enter resnet34 or resnet50')

    if args.gpu:
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    '''
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifa100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: shuffle or not
    Returns: 
        torch dataloader object
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )
    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    '''
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: shuffle or not
    Returns: 
        torch dataloader object
    '''

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return cifar100_test_loader


class WarmUpLR(_LRScheduler):
    '''
    warmup_training learning rate scheduler
    Args:
        optimizer: optimizer(e.g. SGD)
        total_iters: total_iters of warmup phase
    '''
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        Use the first m batches, and set learning rate to
        base_lr * m / total_iters
        '''
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    '''
    return most recent folder under net_weights
    if no non-empty folder were found, return empty folder
    '''
    # get subfolder in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''
    
    # sort folders by folder created time
    folders = sorted(folder, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def best_acc_weights(weights_folder):
    '''
    return the best accuracy .pth file in given folder
    if no, return empty string
    '''
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''
    
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def most_recent_weights(weights_folder):
    '''
    return most recent created weights file
    it not, return empty string
    '''
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''
    
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, keys=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files

def last_epoch(weights_folder):
    '''
    return the most recently epoch
    '''
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('No recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch