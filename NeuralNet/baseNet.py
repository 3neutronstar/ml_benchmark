
def get_hyperparams(model):
    if model == 'lenet5':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif model == 'vgg16':
        dataset = 'cifar10'
        epochs = 300
        lr=1e-2
        momentum=0.9
    elif model=='lenet300_100':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif 'resnet' in model:
        dataset='cifar10'
        lr=1e-2
        epochs=200
        momentum=0.9
    elif model=='convnet':
        dataset = 'cifar10'
        epochs = 200
        lr=1e-2
        momentum=0.9
    elif model=='alexnet':
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
        if 'moo' in configs['mode'] or 'train_lbl' in configs['mode']:
            configs['num_classes']=configs['moo_num_classes']
        else:
            if configs['dataset']=='cifar10' or 'mnist' in configs['dataset']:
                configs['num_classes']=10
            elif configs['dataset']=='cifar100':
                configs['num_classes']=100
            elif configs['dataset']=='imagenet':
                configs['num_classes']=1000

        if configs['model'] == 'lenet5':
            from NeuralNet.lenet5 import LeNet5
            model = LeNet5(configs).to(configs['device'])
        if configs['model'][:3] == 'vgg':
            from NeuralNet.vgg import VGG
            model = VGG(configs).to(configs['device'])
            # print(model)
        if configs['model']=='lenet300_100':
            from NeuralNet.lenet300_100 import LeNet300_100
            model = LeNet300_100(configs).to(configs['device'])
        if configs['model'][:6]=='resnet':
            from NeuralNet.resnet import ResNet
            model = ResNet(configs).to(configs['device'])
        if configs['model']=='convnet':
            from NeuralNet.convnet import ConvNet
            model = ConvNet(configs).to(configs['device'])
        if configs['model']=='alexnet':
            from NeuralNet.alexnet import AlexNet
            model = AlexNet(configs).to(configs['device'])
        self.model=model
