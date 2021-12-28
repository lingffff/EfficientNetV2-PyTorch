# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: For EfficientNet V2 training.
# Create: 2021-12-2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import EfficientNetV2, get_efficientnetv2_params
from eval import eval
from utils import *
from datasets import *



parser = argparse.ArgumentParser(description='Train EfficientNetV2.')
parser.add_argument('model_name', type=str, default='efficientnetv2-b0', 
                    help='name of model')
parser.add_argument('dataset', type=str, default='cifar-10', 
                    help='name of dataset')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.1)


def ajust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return


def train(model, train_loader, criterion, optimizer, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    # for epochs
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        # predict
        preds = model(images)
        loss = criterion(preds, targets)
        # acc
        acc1 = accuracy(preds, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Train: Loss: {losses.avg:.3f}, Acc@1: {top1.avg:.3f}")


def main():
    args = parser.parse_args()
    # prepare model
    num_classes = get_num_classes(args.dataset)
    blocks_args, global_params = get_efficientnetv2_params(args.model_name, num_classes)
    model = EfficientNetV2(blocks_args, global_params)
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # for checkpoint
    # if args.resume:
    #     model = torch.load()
    # prepare dataset
    train_loader, val_loader = get_dataloader(args.dataset, global_params.image_size, args)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    # train
    best_acc1 = 0.0
    for epoch in tqdm(range(args.epochs)):
        ajust_learning_rate(optimizer, epoch, args.lr)
        train(model, train_loader, criterion, optimizer, args)
        torch.save(model, "weights/last.pt")
        acc1 = eval(model, val_loader, criterion, args)
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model, "weights/best.pt")



if __name__ == '__main__':
    main()
