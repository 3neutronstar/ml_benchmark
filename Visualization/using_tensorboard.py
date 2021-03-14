import time
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

class Tensorboard():
    def __init__(self, dataTensor, path, file_name, config):
        if config['nn_type'] == 'lenet5':
            from NeuralNet.lenet5 import w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list
        elif config['nn_type'][:3] == 'vgg':
            from NeuralNet.vgg import get_nn_config
            w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list = get_nn_config(
                config['nn_type'])
        self.w_size_list = w_size_list
        self.b_size_list = b_size_list
        self.NN_size_list = NN_size_list
        self.NN_type_list = NN_type_list
        self.kernel_size_list = kernel_size_list
        self.path=path
        if config['visual_type'] == 'node_domain':
            self.nodeWriter = SummaryWriter(
                log_dir=os.path.join(path,'{}/node_info'.format(file_name)))
        if config['visual_type'] == 'time_domain':
            self.timeWriter = SummaryWriter(
                log_dir=os.path.join(path,'{}/time_info'.format(file_name)))
            self.timeWriter_cum = SummaryWriter(
                log_dir=os.path.join(path,'{}/time_info_cum'.format(file_name)))
        if config['visual_type'] == 'node_domain_integrated':
            # node value integrated for each layer
            self.integratedNodeWriter = SummaryWriter(
                log_dir='visualizing_data/{}/node_xaxis_Integrated'.format(file_name))
        self.total_data = dataTensor
        self.transposed_data = self.total_data.T
        self.nodes_integrated = dict()
        self.time_list = list()


class Tensorboard_node(Tensorboard):  # norm avg기반
    def __init__(self, dataTensor, path, file_name, config):
        super(Tensorboard_node, self).__init__(
            dataTensor, path, file_name, config)
        # dataTensor dim
        # 0: time
        # 1: w 1배
        # 2: avg,norm

        for t, data in enumerate(self.total_data):
            tmp_data = data.detach().clone()
            self.time_list.append(t)
            if t % 100 == 0:
                print('\r {} line complete'.format(t), end='')
            for l, num_w in enumerate(self.b_size_list):  # b인 이유: node관찰이므로
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

        self.info_type_list = [
                               'norm', 'norm_cum', 'var', 'var_cum','avg', 'avg_cum']

        for l, num_node in enumerate(self.b_size_list):
            for n in range(num_node):
                self.nodes_integrated['avg_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['avg_{}l_{}n'.format(l, n)]), dim=0)
                self.nodes_integrated['var_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['var_{}l_{}n'.format(l, n)]), dim=0)
                self.nodes_integrated['norm_cum_{}l_{}n'.format(l, n)] = torch.cumsum(torch.tensor(
                    self.nodes_integrated['norm_{}l_{}n'.format(l, n)]), dim=0)
        print("\nFile Visualization Start")

    def time_write_(self, layer, node, info_type,t):
        # plt.clf()
        # plt.plot(self.time_list, self.nodes_integrated['{}_{}l_{}n'.format(
        #     info_type,layer, node)])
        if 'cum' in info_type:
            self.timeWriter_cum.add_scalar(
                '{}/{}l/{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)
        else:
            self.timeWriter.add_scalar(
                '{}/{}l/{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)

    def time_write(self):
        for info_type in self.info_type_list:
            for l, num_node in enumerate(self.b_size_list):
                for n in range(num_node):
                    print("\rinfo_type: {}_{}l_{}n".format(info_type,l,n),end='')
                    for t in self.time_list:
                        self.time_write_(l, n, info_type,t)
            self.timeWriter.flush()
            self.timeWriter_cum.flush()

        # self.timeWriter.add_scalars(
        #     'avg_of_grads'.format(l), nodes_integrated_avg, t)
        # self.timeWriter.add_scalars(
        #     'norm_of_grads'.format(l), nodes_integrated_norm, t)
        # self.timeWriter.add_scalars(
        #     'var_of_grads'.format(l), nodes_integrated_var, t)
        # self.timeWriter.flush()

        # self.timeWriter_cum.add_scalars(
        #     'avg_of_grads'.format(l), nodes_integrated_avg_cum, t)
        # self.timeWriter_cum.add_scalars(
        #     'norm_of_grads'.format(l), nodes_integrated_norm_cum, t)
        # self.timeWriter_cum.add_scalars(
        #     'var_of_grads'.format(l), nodes_integrated_var_cum, t)
        # self.timeWriter_cum.flush()

    def node_write(self):
        print(self.total_data.size())
        sum_data = torch.mean(self.total_data, dim=0).squeeze(0)
        print(sum_data.size())
        tmp_data = sum_data.detach().clone()
        for l, num_w in enumerate(self.b_size_list):
            node_w = tmp_data[:num_w].detach().clone()
            tmp_data = tmp_data[num_w:]
            for n, node_info in enumerate(node_w):  # node 단위
                self.nodeWriter.add_scalar(
                    'avg_of_grads/{}l'.format(l), node_info[0], n)
                self.nodeWriter.add_scalar(
                    'norm_of_grads/{}l'.format(l), node_info[1], n)
            print('\r {} layer complete'.format(l+1), end='')
            self.nodeWriter.flush()


class Tensorboard_elem(Tensorboard):
    def __init__(self, dataTensor, path, file_name, config):
        super(Tensorboard_elem, self).__init__(
            dataTensor, path, file_name, config)

    def time_write(self):
        # Gradient of node write in time
        # x: time
        # y: sum of grad (each node), norm of grad (each node), norm of grad (each layer)
        nodes_integrated_avg_cum = dict()
        nodes_integrated_norm_cum = dict()
        nodes_integrated_var_cum = dict()
        for t, data in enumerate(self.total_data):
            tmp_data = data.clone().detach()
            if t % 1000 == 0:
                print('\r {} line complete'.format(t), end='')
            for l, (num_w, num_b) in enumerate(zip(self.w_size_list, self.b_size_list)):
                # weight
                tmp_w = tmp_data[:num_w]
                tmp_data = tmp_data[num_w:]  # remove
                nodes_integrated_avg = dict()
                nodes_integrated_norm = dict()
                nodes_integrated_var = dict()
                # self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm in layer(all elem)
                if self.NN_type_list[l] == 'cnn':
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        if t == 0:
                            nodes_integrated_avg_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                            nodes_integrated_norm_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                            nodes_integrated_var_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                        node_w = tmp_w[:(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                        nodes_integrated_avg['{}l_{}n'.format(
                            l, n)] = node_w.sum()
                        nodes_integrated_norm['{}l_{}n'.format(
                            l, n)] = node_w.norm(2)
                        # self.timeWriter.add_scalar('avg_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        # self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w = tmp_w[(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # 내용 제거

                elif self.NN_type_list[l] == 'fc':
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        if t == 0:
                            nodes_integrated_avg_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                            nodes_integrated_norm_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                            nodes_integrated_var_cum['{}l_{}n'.format(
                                l, n)] = 0.0
                        node_w = tmp_w[:self.NN_size_list[l]]
                        nodes_integrated_avg['{}l_{}n'.format(
                            l, n)] = node_w.sum()
                        nodes_integrated_norm['{}l_{}n'.format(
                            l, n)] = node_w.norm(2)
                        nodes_integrated_var['{}l_{}n'.format(
                            l, n)] = node_w.var()
                        nodes_integrated_avg_cum['{}l_{}n'.format(
                            l, n)] += node_w.sum()
                        nodes_integrated_norm_cum['{}l_{}n'.format(
                            l, n)] += node_w.norm(2)
                        nodes_integrated_var_cum['{}l_{}n'.format(
                            l, n)] += node_w.var()
                        # self.timeWriter.add_scalar('avg_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        # self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w = tmp_w[self.NN_size_list[l]:]  # 내용제거
                # bias
                node_b = tmp_data[:num_b].detach().clone()
                tmp_data = tmp_data[num_b:]  # remove
            self.timeWriter.add_scalars(
                'avg_of_grads', nodes_integrated_avg, t)
            self.timeWriter.add_scalars(
                'norm_of_grads', nodes_integrated_norm, t)
            self.timeWriter.add_scalars(
                'var_of_grads', nodes_integrated_var, t)
            self.timeWriter.flush()

            self.timeWriter_cum.add_scalars(
                'avg_of_grads', nodes_integrated_avg, t)
            self.timeWriter_cum.add_scalars(
                'norm_of_grads', nodes_integrated_norm, t)
            self.timeWriter_cum.add_scalars(
                'var_of_grads', nodes_integrated_var_cum, t)
            self.timeWriter_cum.flush()

    def node_write(self):
        sum_time = torch.sum(self.transposed_data, dim=1)
        tmp_data = sum_time.detach().clone()
        for l, (num_w, num_b) in enumerate(zip(self.w_size_list, self.b_size_list)):
            tmp_w = tmp_data[:num_w]  # layer  단위 컷
            tmp_data = tmp_data[num_w:]  # remove

            if self.NN_type_list[l] == 'cnn':
                for n in range(self.NN_size_list[l+1]):  # node 단위
                    node_w = tmp_w[:(
                        self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                    tmp_w = tmp_w[(
                        self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # 내용 제거
                    self.nodeWriter.add_scalar(
                        '{}l/norm_grad'.format(l), node_w.norm(2), n)

            elif self.NN_type_list[l] == 'fc':
                for n in range(self.NN_size_list[l+1]):  # node 단위
                    node_w = tmp_w[:self.NN_size_list[l]]
                    tmp_w = tmp_w[self.NN_size_list[l]:]  # 내용제거
                    self.nodeWriter.add_scalar(
                        '{}l/norm_grad'.format(l), node_w.norm(2), n)

            # bias
            node_b = tmp_data[:num_b].detach().clone()
            tmp_data = tmp_data[num_b:]  # remove
