import os
import sys
import argparse 
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms

from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter 

from conf import settings
from utils import *


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        output = net(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, param in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', param.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', param.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(), 
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(train_dataloader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param, epoch)

    finish = time.time()
    print('epoch {} training time: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_dataloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    
    finish = time.time()
    # if args.gpu:
    #     print('GPU information:')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network...')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        test_loss / len(test_dataloader.dataset),
        correct.float() / len(test_dataloader.dataset),
        finish - start
    ))
    print()

    # add information to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_dataloader.dataset), epoch)

    return correct.float() / len(test_dataloader.dataset)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type, eg: resnet34')
    parser.add_argument('-gpu', action='store_true', default=True, help='use GPU or not')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-num_classes', type=int, default=6, help='number of classes in dataset')
    args = parser.parse_args()

    net = get_network(args)

    train_dir = os.path.join('data', 'seg_train', 'seg_train')
    test_dir = os.path.join('data', 'seg_test', 'seg_test')

    # data preprocessing
    train_dataloader = get_train_data(train_dir, args.batch_size)
    test_dataloader = get_test_data(test_dir, args.batch_size)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) # learning rate decay
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('No recent folder found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # tensorboard cannot overwrite old values
    # the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW
    ))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('Found the best acc weights file: {}'.format(weights_path))
            print('Loading best trainning file to test accuracy...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('Best accuracy is: {:0.2f}'.format(best_acc))
        
        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('No recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('Loading weights file {} to resume training...'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()