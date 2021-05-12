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

    
    data_classes = [i for i in range(configs['moo_num_classes'])]
    random.shuffle(data_classes)
    sparse_data_classes=data_classes[:configs['moo_num_sparse_classes']]
    data_classes.sort()
    sparse_data_classes.sort()

    if configs['moo_num_sparse_classes']==8:
        slice_size=2
    elif configs['moo_num_sparse_classes']==4:
        slice_size=1
    else:
        raise NotImplementedError

    train_data_loader=list()
    test_data_loader=list()
    # for idx in data_classes:
    #     train_subset_dict[ix)] = list()
    #     for j in range(len(train_data)):
    #         if int(train_data[j][1]) == idx:
    #             train_subset_dict[ix)].append(j) # 해당클래스의 인덱스만 추출
    #     locals()['trainset_{}'.format(idx)] = torch.utils.data.Subset(train_data,
    #                                             train_subset_dict[ix)]) # 인덱스 기반 subset 생성

    #     train_data_loader.append(torch.utils.data.DataLoader(locals()['trainset_{}'.format(idx)],
    #                                             batch_size=configs['batch_size'],
    #                                             shuffle=True
    #                                             )) # 각 loader에 넣기
    train_subset_dict=dict()
    for i in data_classes:
        train_subset_dict[i]=list()
    #train
    for idx,(train_images, train_label) in enumerate(train_data):
        if train_label in data_classes:
            train_subset_dict[train_label].append(idx)
        else:
            continue

    min_data_num=min([len(train_subset_dict[i]) for i in data_classes])
    # train data sparsity generator
    for i in data_classes:
        #resize batch size
        if configs['mode']=='train_moo':
            if i in sparse_data_classes:
                batch_size=int(configs['batch_size']*configs['moo_sparse_ratio']/slice_size)
            else:
                batch_size=int(configs['batch_size']/slice_size)
        elif configs['mode']=='baseline_moo':
            batch_size=int(configs['batch_size']/configs['num_classes'])
        else:
            raise NotImplementedError

        # sparse는 줄이기
        if i in sparse_data_classes:
            train_subset_dict[i]=train_subset_dict[i][:int(min_data_num*configs['moo_sparse_ratio'])]
        else:
            train_subset_dict[i]=train_subset_dict[i][:int(min_data_num)]

        # loader에 담기
        locals()['trainset_{}'.format(i)] = torch.utils.data.Subset(train_data,
                                                train_subset_dict[i]) # 인덱스 기반 subset 생성
        train_data_loader.append(torch.utils.data.DataLoader(locals()['trainset_{}'.format(i)],
                                                    batch_size=batch_size,
                                                    pin_memory=pin_memory,
                                                    shuffle=True
                                                    )) # 각 loader에 넣기
        print('{} class have {} data'.format(i,len(train_subset_dict[i])))

    #test
    locals()['test_subset_per_class']=list()
    for idx,(test_images, test_label) in enumerate(test_data):
        if test_label in data_classes:
            locals()['test_subset_per_class'].append(idx)
        else:
            continue

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


    return train_data_loader, test_data_loader
