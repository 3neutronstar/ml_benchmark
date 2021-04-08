import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
def conv_start():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0) # size 줄어듬
        else:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0) # size 줄어들지 않음

        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def deep_make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
    return nn.Sequential(*layers)

#shallow residual
def residual_block(in_dim,out_dim,down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
    layers.extend([
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

class Residual(nn.Module):
    def __init__(self,in_dim,out_dim,down:bool=False,starting:bool=False):
        super(Residual,self).__init__()
        if starting:
            down=False
        layers = []
        if down:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=0))
        else:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        layers.extend([
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        ])
        self.block=nn.Sequential(*layers)
    
        self.relu = nn.ReLU(inplace=True)
        if down:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=1)# size 줄어듬
        else:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=1)# size 줄어들지 않음

        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def shallow_make_layer(in_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Residual(in_dim, out_dim, down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Residual(out_dim, out_dim, down=False))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, configs):
        repeats_dict={'resnet18':[2,2,2,2],'resnet34':[3,4,6,3],'resnet50':[3,4,6,3],'resnet101':[3,4,23,3],'resnet152':[3,8,36,3]}
        repeats=repeats_dict[configs['nn_type']]
        num_classes=configs['num_classes']
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        # 1번
        self.conv1 = conv_start()
        
        # 2번
        base_dim = 64
        if configs['nn_type']=='resnet18' or configs['nn_type']=='resnet34':
            self.conv2 = shallow_make_layer(base_dim, base_dim, repeats[0], starting=True)
            self.conv3 = shallow_make_layer(base_dim, base_dim*2, repeats[1])
            self.conv4 = shallow_make_layer(base_dim*2, base_dim*4, repeats[2])
            self.conv5 = shallow_make_layer(base_dim*4, base_dim*8, repeats[3])
        else:
            self.conv2 = deep_make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
            self.conv3 = deep_make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
            self.conv4 = deep_make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
            self.conv5 = deep_make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])
        
        # 3번
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.classifer = nn.Linear(2048, self.num_classes)

        self.optim=optim.SGD(self.parameters(),configs['lr'],configs['momentum'],nesterov=True)
        self.scheduler=optim.lr_scheduler.StepLR(self.optim,step_size=50,gamma=0.1)
        self.loss=nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        # 3번 2048x1 -> 1x2048
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x