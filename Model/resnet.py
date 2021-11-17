import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def conv3x3(inplanes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, configs):
        super(ResNet, self).__init__()
        type_dict = {'resnet18': ([2, 2, 2, 2], 64, BasicBlock),
                     'resnet34': ([3, 4, 6, 3], 64, BasicBlock),
                     'resnet50': ([3, 4, 6, 3], 64, Bottleneck),
                     'resnet101': ([3, 4, 23, 3], 64, Bottleneck),
                     'resnet152': ([3, 8, 36, 3], 64, Bottleneck),
                     'resnet20': ([3, 3, 3], 16, BasicBlock),
                     'resnet32': ([5, 5, 5], 16, BasicBlock),
                     'resnet44': ([7, 7, 7], 16, BasicBlock),
                     'resnet56': ([9, 9, 9], 16, BasicBlock),
                     'resnet110': ([18, 18, 18], 16, Bottleneck),
                     'resnet1202': ([200, 200, 200], 16, BasicBlock)
                     }
        resnet_type = configs['model']
        num_classes = configs['num_classes']
        self.residual_len = len(type_dict[resnet_type][0])
        if self.residual_len == 4:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.plane_list = [64, 128, 256, 512]
            stride_list = [1, 2, 2, 2]
            self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool=nn.AvgPool2d(7)
        elif self.residual_len == 3:
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.avgpool=nn.AvgPool2d(8)
            self.plane_list = [16, 32, 64]
            stride_list = [1, 2, 2]
        self.device = configs['device']
        block = type_dict[resnet_type][2]
        self.linear = nn.Linear(self.plane_list[-1]*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        residual = list()
        for planes, num_blocks, strides in zip(self.plane_list, type_dict[resnet_type][0], stride_list):
            residual.append(self._make_layer(
                type_dict[resnet_type][2], planes, num_blocks, strides))
        self.residual_layer = nn.Sequential(*residual)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.optim = optim.SGD(params=self.parameters(),
                               momentum=configs['momentum'], lr=configs['lr'], nesterov=configs['nesterov'], weight_decay=configs['weight_decay'])
        self.loss=nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[
                                100, 150], gamma=0.1)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if len(self.plane_list)==4:
            out=self.maxpool(out)
        out = self.residual_layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def extract_feature(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        feature = []
        for residual in self.residual_layer:
            out = residual(out)
            feature.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, feature
