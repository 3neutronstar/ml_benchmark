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
import time
import numpy as np
import matplotlib.pyplot as plt
class BaseLearner():
    def __init__(self,model,time_data,file_path,configs):
        self.model = model.double()
        self.optimizer = self.model.optim
        self.criterion = self.model.loss
        self.scheduler = self.model.scheduler
        self.configs = configs
        self.grad_list = list()
        if configs['mode']=='train_mtl_v2':
            self.log_interval=50
        elif configs['mode']=='train_mtl':
            self.log_interval=50
        else:
            self.log_interval = 50
        self.device = self.configs['device']
        # data
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        self.train_loader, self.test_loader = data_loader(self.configs)
        # Tensorboard
        self.current_path = file_path
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

        # grad list
        self.grad_list = list()

    def save_grad(self, epochs):
        # Save all grad to the file
        self.configs['end_epoch'] = epochs
        if self.configs['grad_save'] == True:
            self.grad_list.append([])
            param_size = list()
            params_write = list()

            tik = time.time()
            if self.configs['nn_type'] == 'lenet300_100' or self.configs['nn_type']=='lenet5':#lenet300 100
                for t, params in enumerate(self.grad_list):
                    if t == 1:
                        for i, p in enumerate(params):  # 각 layer의 params
                            param_size.append(p.size())
                    # elem
                    params_write.append(torch.cat(params, dim=0).unsqueeze(0))
                    # node

                    if t % 100 == 0:
                        print("\r step {} done".format(t), end='')

            # elif self.configs['nn_type'] == 'lenet5': #TODO
            #     for t, params in enumerate(self.grad_list):
            #         if t == 1:
            #             for i, p in enumerate(params):  # 각 layer의 params
            #                 param_size.append(p.size())
            #         # elem
            #         # print(params)
            #         params_write.append(torch.cat(params, dim=0).unsqueeze(0))
            #         # node

            #         if t % 100 == 0:
            #             print("\r step {} done".format(t), end='')

            else:  # vgg16
                import platform
                for epoch in range(1,epochs+1):
                    i = 0
                    epoch_data = list()
                    # check exist
                    while os.path.exists(os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))) == True:
                        batch_idx_data = np.load(os.path.join(
                            self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i)))
                        epoch_data.append(torch.from_numpy(batch_idx_data))
                        i += 1

                    params_write.append(torch.cat(epoch_data, dim=0))
                    print("\r {}epoch processing done".format(epoch),end='')
                print("\n")

            write_data = torch.cat(params_write, dim=0)
            if self.configs['nn_type'] != 'lenet300_100' and self.configs['nn_type']!='lenet5':
                for epoch in range(1,epochs+1):
                    i = 0
                    epoch_data = list()
                    # check exist
                    while os.path.exists(os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))) == True:
                        # remove
                        if platform.system() == 'Windows':
                            os.system('del {}'.format(os.path.join(
                                self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))))
                        else:
                            os.system('rm {}'.format(os.path.join(
                                self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))))
                        i+=1
                    print("\r {}epoch processing done".format(epoch),end='')
            print("\n Write data size:", write_data.size())
            np.save(os.path.join(self.making_path, 'grad_{}'.format(
                self.time_data)), write_data.numpy())  # npy save
            tok = time.time()
            print('play_time for saving:', tok-tik, "s")
            print('size: {}'.format(len(params_write)))

            '''
            Save params
            '''
        return self.configs

    def _save_grad(self, p_groups, epoch, batch_idx):
        # save grad to the list
        if self.configs['grad_save'] == True:
            save_grad_list = list()
            for p in p_groups:
                for l, p_layers in enumerate(p['params']):
                    
                    # node, rest
                    if self.configs['nn_type'] == 'lenet300_100' or self.configs['nn_type']=='lenet5':
                        if len(p_layers.size()) > 1:  # weight filtering
                            p_nodes = p_layers.grad.cpu().detach().clone()
                            # print(p_nodes.size())
                            for n, p_node in enumerate(p_nodes):
                                self.grad_list[-1].append(torch.cat([p_node.mean().view(-1), p_node.norm(
                                ).view(-1), torch.nan_to_num(p_node.var()).view(-1)], dim=0).unsqueeze(0))
                    # elif self.configs['nn_type'] == 'lenet5':#TODO
                    #     if len(p_layers.size()) > 1:  # weight filtering
                    #         p_node = p_layers.grad.view(
                    #             -1).cpu().detach().clone()
                    #         # if i==0:
                    #         #     print(p_node[50:75])
                    #         #     print(p_node.size())
                    #         self.grad_list[-1].append(p_node)

                    else:  # vgg
                        if len(p_layers.size()) > 1:
                            p_nodes = p_layers.grad.cpu().detach().clone()
                            for n, p_node in enumerate(p_nodes):
                                save_grad_list.append(torch.cat([p_node.mean().view(-1), p_node.norm(
                                ).view(-1), torch.nan_to_num(p_node.var()).view(-1)], dim=0).unsqueeze(0))

                    p_layers.to(self.device)
            if 'lenet' not in self.configs['nn_type']:
                npy_path = os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(
                    self.time_data, epoch, batch_idx))
                row_data = torch.cat(save_grad_list, dim=0).unsqueeze(0)
                np.save(npy_path, row_data.numpy())
                del save_grad_list
                del row_data

    def _show_grad(self,output, target,p_groups,epochs,batch_idx):
        if batch_idx%100==0:
            criterion=nn.CrossEntropyLoss(reduction='none')
            self.optimizer.zero_grad()
            loss=criterion(output, target)
            flatten_grads=list()
            for l in loss:
                flatten_grad=list()
                l.backward(retain_graph=True)
                for params in p_groups:
                    for p in params['params']:
                        flatten_grad.append(p.grad.view(-1))
                    flatten_grad=torch.cat(flatten_grad,dim=0)
                flatten_grads.append(flatten_grad.norm().cpu())

            plt.clf()
            plt.plot(flatten_grads,flatten_grads,'bo')
            plt.xlabel('no_pcgrad')
            plt.ylabel('no_pcgrad')
            plt.title('no_pcgrad norm (batch size: {})'.format(self.configs['batch_size']))
            plt.savefig('./grad_data/png/no_pcgrad/{}batch_{}e_{}i.png'.format(self.configs['batch_size'],epochs,batch_idx))
            criterion=nn.CrossEntropyLoss(reduction='mean')
            self.optimizer.zero_grad()
            loss=criterion(output,target)
            loss.backward()
        



         



