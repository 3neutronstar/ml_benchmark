
def get_hyperparams(nn_type):
    if nn_type == 'lenet5':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif nn_type == 'vgg16':
        dataset = 'cifar10'
        epochs = 300
        lr=1e-2
        momentum=0.9
    elif nn_type=='lenet300_100':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif 'resnet' in nn_type:
        dataset='cifar10'
        lr=1e-1
        epochs=200
        momentum=0.9
    elif nn_type=='convnet':
        dataset = 'cifar10'
        epochs = 200
        lr=1e-2
        momentum=0.9
    elif nn_type=='alexnet':
        dataset='cifar10'
        epochs=200
        lr=1e-2
        momentum=0.9
    else:
        print("No algorithm available")
        raise NotImplementedError   

    return dataset,epochs,lr,momentum



class BaseNet():
    def __init__(self,configs):
        
        if configs['dataset'] in ['cifar10','fashionmnist','mnist']:
            configs['num_classes']=10
        elif configs['dataset']=='cifar100':
            configs['num_classes']=100
        else:#imagenet
            configs['num_classes']=1000
        
        if configs['model'] == 'lenet5':
            from Model.lenet5 import LeNet5
            model = LeNet5(configs).to(configs['device'])
        elif configs['model'][:3] == 'vgg':
            from Model.vgg import VGG
            model = VGG(configs).to(configs['device'])
            # print(model)
        elif configs['model']=='lenet300_100':
            from Model.lenet300_100 import LeNet_300_100
            model = LeNet_300_100(configs).to(configs['device'])
        elif configs['model'][:6]=='resnet':
            from Model.resnet import ResNet
            model = ResNet(configs).to(configs['device'])
        elif configs['model']=='convnet':
            from Model.convnet import ConvNet
            model = ConvNet(configs).to(configs['device'])
        elif configs['model']=='alexnet':
            from Model.alexnet import AlexNet
            model = AlexNet(configs).to(configs['device'])
        else:
            print("No Model")
            raise NotImplementedError
        self.model=model

        if 'offkd' in configs['mode'] or 'onkd' in configs['mode']:
            import copy
            KDCONFIGS=copy.deepcopy(configs)
            KDCONFIGS['model']=configs['pretrained_model']
            if configs['pretrained_model'] == 'lenet5':
                from Model.lenet5 import LeNet5
                pretrained_model = LeNet5(KDCONFIGS).to(configs['device'])
            elif configs['pretrained_model'][:3] == 'vgg':
                from Model.vgg import VGG
                pretrained_model = VGG(KDCONFIGS).to(configs['device'])
                # print(pretrained_model)
            elif configs['pretrained_model']=='lenet300_100':
                from Model.lenet300_100 import LeNet_300_100
                pretrained_model = LeNet_300_100(KDCONFIGS).to(configs['device'])
            elif configs['pretrained_model'][:6]=='resnet':
                from Model.resnet import ResNet
                pretrained_model = ResNet(KDCONFIGS).to(configs['device'])
            elif configs['pretrained_model']=='convnet':
                from Model.convnet import ConvNet
                pretrained_model = ConvNet(KDCONFIGS).to(configs['device'])
            elif configs['pretrained_model']=='alexnet':
                from Model.alexnet import AlexNet
                pretrained_model = AlexNet(KDCONFIGS).to(configs['device'])
            else:
                print("No Model")
                raise NotImplementedError
            

            import torch
            if 'offkd' in configs['mode']:
                pretrained_model.load_state_dict(torch.load('./pretrained_data/{}_{}.pth'.format(configs['pretrained_model'],configs['dataset'])))
                self.pretrained_model=pretrained_model.to(configs['device'])