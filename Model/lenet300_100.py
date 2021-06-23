import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LeNet_300_100(nn.Module):

    def __init__(self,configs):
        super(LeNet_300_100, self).__init__()
        self.configs=configs
        self.fc1 = nn.Linear(32*32, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, configs['num_classes'], bias=True)

        self.optim=optim.SGD(params=self.parameters(),momentum=self.configs['momentum'],lr=self.configs['lr'],nesterov=configs['nesterov'],weight_decay=configs['weight_decay'])
        self.scheduler=optim.lr_scheduler.StepLR(self.optim,step_size=15,gamma=0.1)
        self.loss=nn.CrossEntropyLoss()

    def forward(self, x):
        x0 = x.view(-1, 32*32)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3, dim=1)
