import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MaskLayer_F(torch.autograd.Function):
    def forward(self, input, alpha, beta):
        positive = beta.gt(0.8001)
        negative = beta.lt(0.800)
        alpha[positive] = 1.0
        alpha[negative] = 0.0
        # save alpha after we modify it
        self.save_for_backward(input, alpha)
        if not (alpha.eq(1.0).sum() + alpha.eq(0.0).sum() == alpha.nelement()):
            print('ERROR: Please set the weight decay and lr of alpha to 0.0')
        if len(input.shape) == 4:
            input = input.mul(alpha.unsqueeze(2).unsqueeze(3))
        else:
            input = input.mul(alpha)
        return input

    def backward(self, grad_output):
        input, alpha = self.saved_variables
        grad_input = grad_output.clone()
        if len(input.shape) == 4:
            grad_input = grad_input.mul(alpha.data.unsqueeze(2).unsqueeze(3))
        else:
            grad_input = grad_input.mul(alpha.data)

        grad_beta = grad_output.clone()
        grad_beta = grad_beta.mul(input.data).sum(0, keepdim=True)
        if len(grad_beta.shape) == 4:
            grad_beta = grad_beta.sum(3).sum(2)
        return grad_input, None, grad_beta

class MaskLayer(nn.Module):
    def __init__(self, size=-1, conv=False, beta_initial=0.8002, beta_limit=0.802):
        assert(size>0)
        super(MaskLayer, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1, size).zero_().add(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, size).zero_().add(beta_initial))
        self.beta_limit = beta_limit
        self.conv = conv
        return

    def forward(self, x):
        self.beta.data.clamp_(0.0, self.beta_limit)
        x = MaskLayer_F()(x, self.alpha, self.beta)
        return x

class LRN(nn.Module):
    def __init__(self, local_size=1, Alpha=1.0, Beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.Alpha = Alpha
        self.Beta = Beta
        return

    def forward(self, x): 
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.Alpha).add(1.0).pow(self.Beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.Alpha).add(1.0).pow(self.Beta)
        x = x.div(div)
        return x


class ConvNet(nn.Module):
    def __init__(self, configs):
        super(ConvNet, self).__init__()
        self.configs=configs
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = LRN(local_size=3, Alpha=5e-5, Beta=0.75, ACROSS_CHANNELS=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.norm2 = LRN(local_size=3, Alpha=5e-5, Beta=0.75, ACROSS_CHANNELS=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu_conv3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.ip1 = nn.Linear(64*4*4, configs['num_classes'])
        self.optim=optim.SGD(params=self.parameters(),momentum=self.configs['momentum'],lr=self.configs['lr'],nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[
                        100, 150], gamma=0.1)
        self.loss=nn.CrossEntropyLoss()


        self.w_size_list = [5*5*32, 5*5*32, 5*5*64, 64*4*4*configs['num_classes']]  # weight,bias size
        self.b_size_list = [32, 32, 64, configs['num_classes']]
        self.NN_size_list = [3, 32, 32, 64, configs['num_classes']]  # cnn과 fc_net out 작성
        self.NN_type_list = ['cnn', 'cnn', 'cnn', 'fc', 'fc']
        self.kernel_size_list = [(5, 5), (5, 5), (5, 5)]
        self.node_size_list=[6,16,120,84,configs['num_classes']]
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x) 
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu_conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 64*4*4)
        x = self.ip1(x)
        return x

    
    def get_configs(self):
        return self.w_size_list,self.b_size_list,self.NN_size_list,self.NN_type_list,self.node_size_list