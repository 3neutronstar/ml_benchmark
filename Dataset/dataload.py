from numpy import int16
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import sys
from six.moves import urllib
def load_dataset(configs):
    if sys.platform=='linux':
        dataset_path='/data/dataset'
    elif sys.platform=='win32':
        dataset_path='\data\dataset'
    else:
        dataset_path='/data/dataset'
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

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

    if configs['moo_custom']==False:
        data_classes = torch.randperm(dataset_total_num_classes)[:configs['moo_num_classes']].tolist()
        random.shuffle(data_classes)
        sparse_data_classes=data_classes[:configs['moo_num_sparse_classes']]
    else:
        data_classes=configs['moo_custom_class_list']
        sparse_data_classes=configs['moo_sparse_custom_class_list']
        data_classes=[int(d) for d in data_classes]
        sparse_data_classes=[int(sd) for sd in sparse_data_classes]
    data_classes.sort()
    sparse_data_classes.sort()
    print('picked class:',data_classes)
    print('sparse_class:',sparse_data_classes)

    train_data_loader=list()
    test_data_loader=list()
    
    if isinstance(train_data.targets,list):
        # train_data.data=torch.tensor(train_data.data) 
        # test_data.data=torch.tensor(test_data.data) 
        train_data.targets=torch.tensor(train_data.targets) 
        test_data.targets=torch.tensor(test_data.targets) 

    # train data sparsity generator
    idx=torch.zeros_like(train_data.targets)
    print(data_classes)
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

    #sampler setting
    if configs['mode'] in['train_moo','baseline_moo','train_lbl','train_lbl_v2','train_moo_v2','baseline_moo_v2']:#v2 for weighted sum
        sampler=None
        shuffle=True

    # elif configs['mode'] in ['train_moo_v2','baseline_moo_v2']:
    #     from utils import make_weights_for_balanced_classes
    #     sample_weight=make_weights_for_balanced_classes(train_data,nclasses=len(data_classes))       
    #     sampler=torch.utils.data.WeightedRandomSampler(sample_weight,len(sample_weight))
    #     shuffle=False
    else:
        raise NotImplementedError

    train_data_loader=torch.utils.data.DataLoader(train_data,
                                            batch_size=configs['batch_size'],
                                            pin_memory=pin_memory,
                                            shuffle=shuffle,
                                            sampler=sampler
                                            )
    #test
    idx=torch.zeros_like(test_data.targets)
    for predict_idx,class_label in enumerate(data_classes):
        class_idx=(test_data.targets==class_label)
        test_data.targets[class_idx]=predict_idx*torch.ones_like(test_data.targets)[class_idx]# index를 class 맞게 변경
        idx=torch.bitwise_or(idx,class_idx)

    test_data.data=test_data.data[idx.bool()]
    test_data.targets=test_data.targets[idx.bool()] # 인덱스 기반 subset 생성
    test_data_loader=torch.utils.data.DataLoader(test_data,
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
    if configs['mode'] in ['train','train_weight_prune','train_mtl','train_mtl_v2','test','train_cvx']:
        train_data_loader, test_data_loader=base_data_loader(train_data, test_data,configs)
    elif configs['mode']=='train_grad_prune':
        train_data_loader, test_data_loader=split_class_data_loader(train_data, test_data,configs)
    elif 'moo' in configs['mode'] or 'train_lbl' in configs['mode']:
        train_data_loader, test_data_loader=split_class_list_data_loader(train_data, test_data,configs)
    else:
        raise NotImplementedError



    return train_data_loader, test_data_loader
