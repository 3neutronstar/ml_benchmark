from torchvision import datasets
import torchvision.transforms as transforms
import torch


def load_dataset(configs):
    if configs['dataset'] == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False,
                                        download=False, transform=transform)

    elif configs['dataset'] == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR100(root='data', train=True,
                                       download=True, transform=transform)
        test_data = datasets.CIFAR100(root='data', train=False,
                                      download=False, transform=transform)

    elif configs['dataset'] == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10(root='data', train=True,
                                      download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False,
                                     download=False, transform=transform)

    return train_data, test_data


def data_loader(configs):
    train_data, test_data = load_dataset(configs)
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
