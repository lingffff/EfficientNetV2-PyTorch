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
            train_set = torchvision.datasets.CIFAR10(ROOT, train=True, transform=train_transforms, download=True)
            val_set = torchvision.datasets.CIFAR10(ROOT, train=False, transform=val_transforms, download=True)
        elif dataset == 'cifar100':
            train_set = torchvision.datasets.CIFAR100(ROOT, train=True, transform=train_transforms, download=True)
            val_set = torchvision.datasets.CIFAR100(ROOT, train=False, transform=val_transforms, download=True)
 
    
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
    
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers)
    if train:
        return train_loader, val_loader
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
