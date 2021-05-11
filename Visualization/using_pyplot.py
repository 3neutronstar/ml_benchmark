import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import math


class Pyplot_node():
    def __init__(self, fileTensor, config, path,file_name):
        NUM_ROWS = config['epochs'] * \
            math.ceil(60000.0/float(config['batch_size']))
        if config['model'] == 'lenet5':
            from NeuralNet.lenet5 import w_size_list, b_size_list, NN_size_list, model_list, kernel_size_list
        elif config['model'][:3] == 'vgg':
            from NeuralNet.vgg import get_nn_config
            w_size_list, b_size_list, NN_size_list, model_list, kernel_size_list = get_nn_config(
                config['model'])
        if os.path.exists(os.path.join(path,file_name)) == False:
            os.mkdir(os.path.join(path,file_name))  
        dir_list=['node_info','node_integrated_info']
        for dir in dir_list:
            if os.path.exists(os.path.join(path,file_name,dir)) == False:
                os.mkdir(os.path.join(path,file_name,dir))  

        self.path = path
        self.file_name=file_name
        self.w_size_list = w_size_list
        self.b_size_list = b_size_list
        self.NN_size_list = NN_size_list
        self.model_list = model_list
        self.kernel_size_list = kernel_size_list
        self.nodes_integrated = dict()
        total_data = fileTensor.clone()
        self.time_list = list()
        for t, data in enumerate(total_data):
            tmp_data = data.detach().clone()
            self.time_list.append(t)
            if t % 100 == 0:
                print('\r {} line complete'.format(t), end='')
            for l, num_w in enumerate(b_size_list):  # b인 이유: node관찰이므로
                # weight
                node_w = tmp_data[:num_w].detach().clone()
                tmp_data = tmp_data[num_w:]
                for n, node_info in enumerate(node_w):  # node 단위
                    if t == 0:
                        self.nodes_integrated['avg_{}l_{}n'.format(
                            l, n)] = list()
                        self.nodes_integrated['norm_{}l_{}n'.format(
                            l, n)] = list()
                        self.nodes_integrated['var_{}l_{}n'.format(
                            l, n)] = list()

                    self.nodes_integrated['avg_{}l_{}n'.format(
                        l, n)].append(node_info[0])
                    self.nodes_integrated['norm_{}l_{}n'.format(
                        l, n)].append(node_info[1])
                    self.nodes_integrated['var_{}l_{}n'.format(
                        l, n)].append(node_info[2])
        print("File Visualization Start")

        self.info_type_list = ['avg', 'avg_cum',
                               'norm', 'norm_cum', 'var', 'var_cum']

    def time_write_(self, layer, node, info_type):
        plt.clf()
        plt.plot(
            self.time_list, torch.tensor(self.nodes_integrated['{}_{}l_{}n'.format(info_type, layer, node)]).tolist())
        plt.xlabel('iter')
        plt.ylabel('{} of grad in node'.format(info_type))
        plt.savefig(os.path.join(self.path,self.file_name,
                                 'node_info', '{}_{}l_{}n.png'.format(info_type, layer, node)), dpi=100)

    def time_write(self):
        for l, num_node in enumerate(self.b_size_list):
            for n in range(num_node):
                self.nodes_integrated['avg_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['avg_{}l_{}n'.format(l, n)]), dim=0)
                self.nodes_integrated['var_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['var_{}l_{}n'.format(l, n)]), dim=0)
                self.nodes_integrated['norm_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['norm_{}l_{}n'.format(l, n)]), dim=0)
                for info_type in self.info_type_list:
                    self.time_write_(l, n, info_type)

        # integrated value plot
    def time_integrated_write_(self, num_node, layer, info_type):
        plt.clf()
        legend_list = list()
        for n in range(num_node):
            plt.plot(
                self.time_list, self.nodes_integrated['{}_{}l_{}n'.format(info_type, layer, n)])
            legend_list.append(['{}l_{}n'.format(layer, n)])
        plt.xlabel('iter')
        plt.ylabel('avg of grad in node')
        plt.legend(legend_list)
        plt.savefig(os.path.join(self.path,self.file_name,
                                 'node_integrated_info', '{}_{}l_{}n.png'.format(layer, n)), dpi=150)
        del legend_list

    def time_integrated_write(self):
        for l, num_node in enumerate(self.b_size_list):
            for info_type in self.info_type_list:
                self.time_integrated_write_(num_node, l, info_type)
