from torchvision import datasets
import torchvision.transforms as transforms
import torch
def load_dataset(configs):
    if configs['dataset']=='mnist':
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()]))
        test_data = datasets.MNIST(root='data', train=False,
                                        download=False, transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
    
    return train_data,test_data

def data_loader(configs):
    train_data,test_data=load_dataset(configs)
    if configs['device']=='gpu':
        pin_memory=True
    else:
        pin_memory=False
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=configs['batch_size'],
                                          shuffle=True,
                                          pin_memory=pin_memory,
                                        )
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=configs['batch_size'],
                                          shuffle=True,
                                          pin_memory=pin_memory,
                                        )

    print("Using Datasets: ",configs['dataset'])
    return train_data_loader,test_data_loader