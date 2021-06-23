import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, configs):
        super(ResNet, self).__init__()
        type_dict={'resnet18':([2,2,2,2],64,BasicBlock),
        'resnet34':([3,4,6,3],64,BasicBlock),
        'resnet50':([3,4,6,3],64,Bottleneck),
        'resnet101':([3,4,23,3],64,Bottleneck),
        'resnet152':([3,8,36,3],64,Bottleneck),
        'resnet20':([3,3,3],16,BasicBlock),
        'resnet32':([5,5,5],16,BasicBlock),
        'resnet44':([7,7,7],16,BasicBlock),
        'resnet56':([9,9,9],16,BasicBlock),
        'resnet110':([18,18,18],16,BasicBlock),
        'resnet1202':([200,200,200],16,BasicBlock)
        }
        resnet_type=configs['model']
        num_classes=configs['num_classes']
        self.residual_len=len(type_dict[resnet_type][0])
        if self.residual_len==4:
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            plane_list=[64,128,256,512]
            stride_list=[1,2,2,2]
        elif self.residual_len==3:
            self.in_planes=16
            self.in_planes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            plane_list=[16,32,64]
            stride_list=[1,2,2]
        self.device=configs['device']        
        block=type_dict[resnet_type][2]
        self.linear = nn.Linear(plane_list[-1]*block.expansion, num_classes)
        residual=list()
        for planes,num_blocks,strides in zip(plane_list,type_dict[resnet_type][0],stride_list):
            residual.append(self._make_layer(type_dict[resnet_type][2],planes,num_blocks,strides))
        self.residual_layer=nn.Sequential(*residual)
        self.optim = optim.SGD(params=self.parameters(),
                               momentum=configs['momentum'], lr=configs['lr'], nesterov=configs['nesterov'], weight_decay=configs['weight_decay'])
        self.loss=nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[
                                100, 150], gamma=0.1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride).to(self.device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.residual_layer(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
