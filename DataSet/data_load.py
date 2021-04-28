from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torchvision.transforms.transforms import RandomCrop


def load_dataset(configs):
    if configs['dataset'] == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False,
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
        train_data = datasets.CIFAR100(root='data', train=True,
                                       download=True, transform=train_transform)
        test_data = datasets.CIFAR100(root='data', train=False,
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
        
        train_data = datasets.CIFAR10(root='data', train=True,
                                      download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root='data', train=False,
                                     download=False, transform=test_transform)

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
    data_classes = [i for i in range(configs['num_classes'])]
    train_data_loader=list()
    # for idx in data_classes:
    #     locals()['train_subset_per_class_{}'.format(idx)] = list()
    #     for j in range(len(train_data)):
    #         if int(train_data[j][1]) == idx:
    #             locals()['train_subset_per_class_{}'.format(idx)].append(j) # 해당클래스의 인덱스만 추출
    #     locals()['trainset_{}'.format(idx)] = torch.utils.data.Subset(train_data,
    #                                             locals()['train_subset_per_class_{}'.format(idx)]) # 인덱스 기반 subset 생성

    #     train_data_loader.append(torch.utils.data.DataLoader(locals()['trainset_{}'.format(idx)],
    #                                             batch_size=configs['batch_size'],
    #                                             shuffle=True
    #                                             )) # 각 loader에 넣기
    for i in range(configs['num_classes']):
        locals()['train_subset_per_class_{}'.format(i)]=list()
    for idx,(images, label) in enumerate(train_data):
        locals()['train_subset_per_class_{}'.format(label)].append(idx)
    for i in range(configs['num_classes']):
        locals()['trainset_{}'.format(i)] = torch.utils.data.Subset(train_data,
                                                locals()['train_subset_per_class_{}'.format(i)]) # 인덱스 기반 subset 생성
        train_data_loader.append(torch.utils.data.DataLoader(locals()['trainset_{}'.format(i)],
                                                batch_size=configs['batch_size'],
                                                shuffle=True
                                                )) # 각 loader에 넣기
        
    test_data_loader = torch.utils.data.DataLoader(test_data,
                    batch_size=configs['batch_size'], shuffle=False)
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
    if configs['mode']=='train' or configs['mode']=='train_weight_prune' or configs['mode']=='train_mtl' or configs['mode']=='train_mtl_v2':
        train_data_loader, test_data_loader=base_data_loader(train_data, test_data,configs)
    elif configs['mode']=='train_grad_prune':
        train_data_loader, test_data_loader=split_class_data_loader(train_data, test_data,configs)
    # elif configs['mode']=='train_mtl_v4':# Not Using now
    #     train_data_loader, test_data_loader=split_class_list_data_loader(train_data, test_data,configs)


    return train_data_loader, test_data_loader
