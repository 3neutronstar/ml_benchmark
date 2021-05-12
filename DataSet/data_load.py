from numpy import int16
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import sys

def load_dataset(configs):
    if sys.platform=='linux':
        dataset_path='/dataset'
    elif sys.platform=='win32':
        dataset_path='\dataset'
    else:
        dataset_path='./dataset'

    if configs['dataset'] == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
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

def split_class_data_loader(train_data,test_data,configs):
    if configs['dataset']=='mnist' or configs['dataset']=='cifar10' or configs['cifar100']:
        data_classes = [i for i in range(configs['num_'])] # MNIST
        idx =1 # split class index
        locals()['train_subset_per_class_{}'.format(1)] = list()
        for j in range(len(train_data)):
            if int(train_data[j][1]) == idx:#index 확인
                locals()['train_subset_per_class_{}'.format(idx)].append(j)
        locals()['trainset_{}'.format(idx)] = torch.utils.data.Subset(train_data,
                                                locals()['train_subset_per_class_{}'.format(idx)])

        train_data_loader = torch.utils.data.DataLoader(locals()['trainset_{}'.format(idx)],
                                                batch_size=configs['batch_size'],
                                                shuffle=True
                                                )

        test_data_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=configs['batch_size'], shuffle=False)

    return train_data_loader, test_data_loader


def split_class_list_data_loader(train_data,test_data,configs):
    import random
    
    if configs['device'] == 'gpu':
        pin_memory = True
    else:
        pin_memory = False
    if configs['dataset']=='cifar10' or 'mnist' in configs['dataset']:
        dataset_total_num_classes=10
    elif configs['dataset']=='cifar100':
        dataset_total_num_classes=100
    elif configs['dataset']=='imagenet':
        dataset_total_num_classes=1000

    data_classes = torch.randperm(dataset_total_num_classes)[:configs['moo_num_classes']].tolist()
    random.shuffle(data_classes)
    sparse_data_classes=data_classes[:configs['moo_num_sparse_classes']]
    data_classes.sort()
    sparse_data_classes.sort()
    print('picked class:',data_classes)
    print('sparse_class:',sparse_data_classes)

    train_data_loader=list()
    test_data_loader=list()
    
    # train data sparsity generator
    idx=torch.zeros_like(train_data.targets)
    for predict_idx,class_label in enumerate(data_classes):
        if class_label in sparse_data_classes:
            non_zero_idx=torch.nonzero(train_data.targets==class_label)
            class_size=non_zero_idx.size()[0]
            non_zero_idx=non_zero_idx[torch.randperm(class_size)][:int(class_size*configs['moo_sparse_ratio'])]
            class_idx=torch.zeros_like(train_data.targets==class_label)
            class_idx[non_zero_idx]=1
        else:
            class_idx=(train_data.targets==class_label)
        train_data.targets[class_idx]=predict_idx*torch.ones_like(train_data.targets)[class_idx]# index를 class 맞게 변경
        idx=torch.bitwise_or(idx,class_idx)
    train_data.data=train_data.data[idx.bool()]
    train_data.targets=train_data.targets[idx.bool()]

    train_data_loader=torch.utils.data.DataLoader(train_data,
                                            batch_size=configs['batch_size'],
                                            pin_memory=pin_memory,
                                            shuffle=True
                                            )
    #test
    locals()['test_subset_per_class']=list()
    for idx,(test_images, test_label) in enumerate(test_data):
        if test_label in data_classes:
            locals()['test_subset_per_class'].append(idx)
        else:
            continue

    for predict_idx,class_label in enumerate(data_classes):
        class_idx=(test_data.targets==class_label)
        test_data.targets[class_idx]=predict_idx*torch.ones_like(test_data.targets)[class_idx]# index를 class 맞게 변경

    locals()['testset'] = torch.utils.data.Subset(test_data,
                                            locals()['test_subset_per_class']) # 인덱스 기반 subset 생성
    test_data_loader=torch.utils.data.DataLoader(locals()['testset'],
                                            batch_size=configs['batch_size'],
                                            pin_memory=pin_memory,
                                            shuffle=False
                                            ) # 각 loader에 넣기
    print("Finish Load splitted dataset")
    return train_data_loader, test_data_loader #list(loader),loader return


def base_data_loader(train_data,test_data,configs):
    if configs['device'] == 'gpu':
        pin_memory = True
        # pin_memory=False
    else:
        pin_memory = False
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=configs['batch_size'],
                                                    shuffle=True,
                                                    pin_memory=pin_memory,
                                                    num_workers=configs['num_workers'],
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=configs['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=pin_memory,
                                                   num_workers=configs['num_workers'],
                                                   )

    print("Using Datasets: ", configs['dataset'])
    return train_data_loader, test_data_loader

def data_loader(configs):
    train_data, test_data = load_dataset(configs)
    if configs['mode']=='train' or configs['mode']=='train_weight_prune' or configs['mode']=='train_mtl' or configs['mode']=='train_mtl_v2' or configs['mode']=='test':
        train_data_loader, test_data_loader=base_data_loader(train_data, test_data,configs)
    elif configs['mode']=='train_grad_prune':
        train_data_loader, test_data_loader=split_class_data_loader(train_data, test_data,configs)
    elif 'moo' in configs['mode']:
        train_data_loader, test_data_loader=split_class_list_data_loader(train_data, test_data,configs)
    else:
        raise NotImplementedError



    return train_data_loader, test_data_loader
