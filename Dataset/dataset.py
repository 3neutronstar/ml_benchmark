from numpy import int16
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import sys
from six.moves import urllib
from Dataset.preprocess import *
def load_dataset(configs):
    if sys.platform=='linux':
        dataset_path='../data/dataset'
    elif sys.platform=='win32':
        dataset_path='..\data\dataset'
    else:
        dataset_path='../data/dataset'

    
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    if configs['dataset'] == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=dataset_path, train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=dataset_path, train=False,
                                        download=False, transform=transform)

    elif configs['dataset'] == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_data = datasets.CIFAR100(root=dataset_path, train=True,
                                       download=True, transform=train_transform)
        test_data = datasets.CIFAR100(root=dataset_path, train=False,
                                      download=False, transform=test_transform)

    elif configs['dataset'] == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        train_transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_data = datasets.CIFAR10(root=dataset_path, train=True,
                                      download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root=dataset_path, train=False,
                                     download=False, transform=test_transform)
    
    elif configs['dataset']=='fashionmnist':
        train_transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data=datasets.FashionMNIST(root=dataset_path, download=True, train=True, transform=train_transform)
        test_data=datasets.FashionMNIST(root=dataset_path, download=False, train=False, transform=test_transform)
    
    else:
        raise NotImplementedError

    return train_data, test_data
