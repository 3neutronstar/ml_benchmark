
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
        
        if configs['dataset']=='cifar10' or configs['dataset']=='mnist':
            configs['num_classes']=10
        else:
            configs['num_classes']=100

        if configs['nn_type'] == 'lenet5':
            from NeuralNet.lenet5 import LeNet5
            model = LeNet5(configs).to(configs['device'])
        if configs['nn_type'][:3] == 'vgg':
            from NeuralNet.vgg import VGG
            model = VGG(configs).to(configs['device'])
            # print(model)
        if configs['nn_type']=='lenet300_100':
            from NeuralNet.lenet300_100 import LeNet_300_100
            model = LeNet_300_100(configs).to(configs['device'])
        if configs['nn_type'][:6]=='resnet':
            from NeuralNet.resnet import ResNet
            model = ResNet(configs).to(configs['device'])
        if configs['nn_type']=='convnet':
            from NeuralNet.convnet import ConvNet
            model = ConvNet(configs).to(configs['device'])
        if configs['nn_type']=='alexnet':
            from NeuralNet.alexnet import AlexNet
            model = AlexNet(configs).to(configs['device'])
        self.model=model