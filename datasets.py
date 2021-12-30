import os
import torch
import torchvision
import torchvision.transforms as transforms

ROOT = './data'

def get_dataloader(dataset: str, img_size, args, train=True):
    if "cifar" in dataset:
        normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])
        if dataset == 'cifar10':
            train_set = torchvision.datasets.CIFAR10(ROOT, train=True, transform=train_transforms)
            val_set = torchvision.datasets.CIFAR10(ROOT, train=False, transform=val_transforms)
        elif dataset == 'cifar100':
            train_set = torchvision.datasets.CIFAR100(ROOT, train=True, transform=train_transforms)
            val_set = torchvision.datasets.CIFAR100(ROOT, train=False, transform=val_transforms)
 
    
    elif dataset == 'imagenet':
        traindir = os.path.join(ROOT, 'train')
        valdir = os.path.join(ROOT, 'val')

        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_set = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ]))
        val_set = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalization,
            ]))
    else:
        raise NotImplementedError
    
    train_sampler = None
    if hasattr(args, 'ddp'):
        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    if train:
        return train_loader, val_loader, train_sampler
    else:
        return val_loader


def get_num_classes(dataset: str):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    else:
        raise NotImplementedError
