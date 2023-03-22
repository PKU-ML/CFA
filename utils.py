import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
from torchvision import datasets, transforms
import numpy as np

PATH = '/home/kemove/data/cifar_data'
def dev(id):
    return f'cuda:{id}'

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std =  (0.2675, 0.2565, 0.2761)

mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

mu_100 = torch.tensor(cifar100_mean).view(3,1,1)
std_100 = torch.tensor(cifar100_std).view(3,1,1)

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

def normalize_cifar_100(x):
    return (x - mu_100.to(x.device))/(std_100.to(x.device))

def load_dataset(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        transform_ = transforms.Compose([transforms.ToTensor()])
        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(PATH, train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    elif dataset == 'cifar100':
        transform_ = transforms.Compose([transforms.ToTensor()])

        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(PATH, train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(PATH, train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

def load_valid_dataset(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        train_loader, valid_loader = torch.load('data/split_dataset.pth')

        # test loader
        transform_ = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    elif dataset == 'cifar100':
        train_loader, valid_loader = torch.load('data/split_dataset_100.pth')

        # test loader
        transform_ = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(PATH, train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

def load_cw_dataset(dataset='cifar10', batch_size=128, valid=True):
    if dataset == 'cifar10':
        if valid:
            train_loader, valid_loader = torch.load('cifar_data/split_dataset.pth')
        else:
            train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
            train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(PATH, train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        
        # test loader
        transform_ = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)
        data = torch.cat([x for (x,y) in test_loader], dim=0)
        label = torch.cat([y for (x,y) in test_loader], dim=0)

        cw_test_loader = []
        for i in range(10):
            index = (label==i).nonzero().flatten()
            loader = []
            for j in range(10):
                curr_index = index[j*100:(j+1)*100]
                loader.append((data[curr_index], label[curr_index]))
            cw_test_loader.append(loader)

        if valid:
            return train_loader, valid_loader, cw_test_loader
        else:
            return train_loader, cw_test_loader

def weight_average(model, new_model, decay_rate, init=False):
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    if init:
        decay_rate = 0
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict)

