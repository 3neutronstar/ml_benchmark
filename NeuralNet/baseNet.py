
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
        self.model=model