import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5)) # 5x5+1 params
        self.subsampling=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5)) # 5x5+1 params
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5)) # 5x5+1 params
        self.fc1=nn.Linear(120,84)
        self.fc2=nn.Linear(84,10)
        self.log_softmax=nn.LogSoftmax(dim=-1)
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.subsampling(x)
        x=F.relu(self.conv2(x))
        x=self.subsampling(x)
        x=F.relu(self.conv3(x))
        x=x.view(-1,120)
        x=F.relu(self.fc1(x))
        x=self.log_softmax(x)
        return x

