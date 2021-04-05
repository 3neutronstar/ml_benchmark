
class BaseNet():
    def __init__(self,configs):
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
            configs['num_classes']=10
            model = ResNet(configs).to(configs['device'])
        self.model=model