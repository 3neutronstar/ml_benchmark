import torch.nn as nn
import torch.optim as optim

class AlexNet(nn.Module):
    def __init__(self, configs):
        super(AlexNet, self).__init__()
        self.configs=configs
        self.prune = False
        self.num_classes=configs['num_classes']

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        self.optim = optim.SGD(params=self.parameters(),
                               momentum=configs['momentum'], lr=configs['lr'], nesterov=configs['nesterov'], weight_decay=configs['weight_decay'])

        self.loss=nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[
                                100, 150], gamma=0.1)
                                
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x