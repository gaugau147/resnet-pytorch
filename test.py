import argparse

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import *

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='network type, e.g. resnet34')
    parser.add_argument('-weights', type=str, required=True, help='the weight file')
    parser.add_argument('-gpu', type=bool, default=True, help='use GPU or not')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-num_classes', type=int, default=6, help='number of classes in dataset')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print('iteration: {}\ttotal {} iterations'.format(n_iter+1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top 1
            correct_1 += correct[:, :1].sum()

    print()
    print('Top 1 error: ', 1 - correct_1 / len(cifar100_test_loader.dataset))
    print('Top 5 error: ', 1 - correct_5 / len(cifar100_test_loader))
    print('Parameter numbers: {}'.format(sum(p.numel() for p in net.parameters())))
