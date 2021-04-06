import torch
from DataSet.data_load import data_loader

import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from six.moves import urllib
import os
from utils import EarlyStopping

class BaseLearner():
    def __init__(self,model,time_data,configs):
        self.model = model
        self.optimizer = self.model.optim
        self.criterion = self.model.loss
        self.scheduler = self.model.scheduler
        self.configs = configs
        self.grad_list = list()
        self.log_interval = 100
        self.device = self.configs['device']
        # data
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        self.train_loader, self.test_loader = data_loader(self.configs)
        # Tensorboard
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.logWriter = SummaryWriter(os.path.join(
            self.current_path, 'training_data', time_data))
        self.time_data = time_data

        self.early_stopping = EarlyStopping(
            self.current_path, time_data, configs, patience=self.configs['patience'], verbose=True)
        if self.configs['colab'] == True:
            self.making_path = os.path.join('drive', 'MyDrive', 'grad_data')
        else:
            self.making_path = os.path.join(self.current_path, 'grad_data')
        if os.path.exists(self.making_path) == False:
            os.mkdir(self.making_path)
        if os.path.exists(os.path.join(self.making_path, 'tmp')) == False:
            os.mkdir(os.path.join(self.making_path, 'tmp'))