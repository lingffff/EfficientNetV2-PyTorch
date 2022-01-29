# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: For EfficientNet V2 training.
# Create: 2021-12-2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import shutil
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import EfficientNetV2, get_efficientnetv2_params
from eval import eval
from utils import *
from datasets import *



parser = argparse.ArgumentParser(description='Train EfficientNetV2.')
parser.add_argument('model_name', type=str, default='efficientnetv2-b0', 
                    help='name of model')
parser.add_argument('dataset', type=str, default='cifar10', 
                    help='name of dataset')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--ddp', action='store_true', 
                    help='Distributed Data Parallel Training on ONE server.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)


def ajust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return


def save_checkpoint(state, is_best, dir='weights'):
    last = os.path.join(dir, 'last.pth.tar')
    torch.save(state, last)
    if is_best:
        best = os.path.join(dir, 'best.pth.tar')
        shutil.copyfile(last, best)


def train(model, train_loader, criterion, optimizer, rank):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    # for epochs
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda(rank)
        targets = targets.cuda(rank)
        # predict
        preds = model(images)
        loss = criterion(preds, targets)
        # acc
        acc1 = accuracy(preds, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Train: Loss: {losses.avg:.3f}, Acc@1: {top1.avg:.3f}")
    return losses.avg, top1.avg


def main():
    args = parser.parse_args()
    # lucky seed!
    set_seed(42)

    if args.ddp:
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    else:
        main_worker(None, None, args)

best_acc1 = 0.0
writer = SummaryWriter('./runs')

def main_worker(rank, world_size, args):
    global best_acc1
    if args.ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        # initialize the process group
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    # prepare model
    num_classes = get_num_classes(args.dataset)
    blocks_args, global_params = get_efficientnetv2_params(args.model_name, num_classes)
    model = EfficientNetV2(blocks_args, global_params)
    if args.ddp:
        torch.cuda.set_device(rank)
        model.cuda(rank)
        args.batch_size = int(args.batch_size / world_size)
        args.workers = int(args.workers / world_size)
        print(f"==> use GPU {rank} for DDP training.")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model.cuda()
        if torch.cuda.device_count() > 1:
            print(f"==> use DP for training.")
            model = nn.DataParallel(model)
    # prepare dataset
    train_loader, val_loader, train_sampler = get_dataloader(args.dataset, global_params.image_size, args)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # for checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if rank is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(rank)
            print(f"=> start_epoch = {start_epoch}, best_acc1 = {best_acc1:.2f}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    # start train
    for epoch in tqdm(range(start_epoch, args.epochs)):
        if args.ddp:
            train_sampler.set_epoch(epoch)
        ajust_learning_rate(optimizer, epoch, args.lr)
        train_loss, train_acc1 = train(model, train_loader, criterion, optimizer, rank)
        val_loss, val_acc1 = eval(model, val_loader, criterion, rank)
        
        if not args.ddp or (args.ddp and rank == 0):
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc1/train', train_acc1, epoch)
            writer.add_scalar('Acc1/val', val_acc1, epoch)

        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
        if not args.ddp or (args.ddp and rank == 0):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'arch': args.model_name,
                'epoch': epoch,
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict()
                }, is_best)



if __name__ == '__main__':
    main()
