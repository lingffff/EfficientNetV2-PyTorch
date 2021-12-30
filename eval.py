# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: For EfficientNet V2 evaluation.
# Create: 2021-12-2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch.nn as nn

from model import EfficientNetV2, get_efficientnetv2_params
from utils import *
from datasets import *



parser = argparse.ArgumentParser(description='Test EfficientNetV2.')
parser.add_argument('model_name', type=str, default='efficientnetv2-b0', 
                    help='name of model')
parser.add_argument('dataset', type=str, default='cifar-10', 
                    help='name of dataset')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--ckpt', type=str, default='', required=True)


def eval(model, val_loader, criterion, rank):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    # for epochs
    for i, (images, targets) in enumerate(val_loader):
        images = images.cuda(rank)
        targets = targets.cuda(rank)
        # predict
        preds = model(images)
        loss = criterion(preds, targets)
        # acc
        acc1 = accuracy(preds, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

    print(f"[Test] >>>>>>>>>> Loss: {losses.avg:.3f}, Acc@1: {top1.avg:.3f}")
    return top1.avg


def main():
    args = parser.parse_args()
    # prepare params
    num_classes = get_num_classes(args.dataset)
    blocks_args, global_params = get_efficientnetv2_params(args.model_name, num_classes)
    # prepare model
    model = EfficientNetV2(blocks_args, global_params)
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # load model
    if os.path.isfile(args.ckpt):
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        best_acc1 = checkpoint['best_acc1']
        print(f">>> best Acc@1 saved in ckpt: {best_acc1:.3f}")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"=> no checkpoint found at '{args.ckpt}'")
    # prepare dataset
    val_loader = get_dataloader(args.dataset, global_params.image_size, args, train=False)
    # test
    criterion = nn.CrossEntropyLoss().cuda()
    eval(model, val_loader, criterion, None)



if __name__ == '__main__':
    main()
