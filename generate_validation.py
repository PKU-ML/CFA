PATH = 'data/cifar_data' #YOUR DATAPATH HERE

import torch
import argparse
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()
dataset = args.dataset

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')
    if dataset == 'cifar10':
    # CIFAR10
        train_transform_ = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        transform_ = transforms.Compose([transforms.ToTensor()])        
        ori_set = datasets.CIFAR10(PATH, train=True, download=True, transform=train_transform_)
        valid_set = datasets.CIFAR10(PATH, train=True, download=True, transform=transform_)
        ori_label = torch.tensor([y for (_, y) in ori_set])
        n = 100 # for each classes (2% of 5000)
        valid_index, train_index = [], []
        for i in range(10):
            valid_index_i = (ori_label==i).nonzero()[:n]
            train_index_i = (ori_label==i).nonzero()[n:]
            valid_index.append(valid_index_i)
            train_index.append(train_index_i)
        valid_index = torch.cat(valid_index, dim=0).flatten()
        train_index = torch.cat(train_index, dim=0).flatten()

        N = len(train_index)
        order = np.random.permutation(N)
        train_index = train_index[order]
        
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        train_loader = torch.utils.data.DataLoader(ori_set, batch_size=128, shuffle=False, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False, sampler=valid_sampler)

        torch.save([train_loader, valid_loader], 'data/split_dataset.pth')
        
    else:
        # CIFAR 100
        train_transform_ = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        transform_ = transforms.Compose([transforms.ToTensor()])        
        ori_set = datasets.CIFAR100(PATH, train=True, download=True, transform=train_transform_)
        valid_set = datasets.CIFAR100(PATH, train=True, download=True, transform=transform_)
        ori_label = torch.tensor([y for (_, y) in ori_set])
        n = 25 # for each classes
        valid_index, train_index = [], []
        for i in range(100):
            valid_index_i = (ori_label==i).nonzero()[:n]
            train_index_i = (ori_label==i).nonzero()[n:]
            valid_index.append(valid_index_i)
            train_index.append(train_index_i)
        valid_index = torch.cat(valid_index, dim=0).flatten()
        train_index = torch.cat(train_index, dim=0).flatten()

        N = len(train_index)
        order = np.random.permutation(N)
        train_index = train_index[order]
        
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        train_loader = torch.utils.data.DataLoader(ori_set, batch_size=128, shuffle=False, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False, sampler=valid_sampler)

        torch.save([train_loader, valid_loader], 'data/split_dataset_100.pth')
        
