import time
import torch
from torch.utils.tensorboard import SummaryWriter


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
        if config['visual_type'] == 'node_domain':
            self.nodeWriter = SummaryWriter(
                log_dir='visualizing_data/{}/node_domain(x)'.format(file_name),)
        elif config['visual_type'] == 'time_domain':
            self.timeWriter = SummaryWriter(
                log_dir='visualizing_data/{}/time_domain(x)'.format(file_name))
        elif config['visual_type'] == 'node_domain_integrated':
            # node value integrated for each layer
            self.integratedNodeWriter = SummaryWriter(
                log_dir='visualizing_data/{}/node_domain(x)_Integrated'.format(file_name))
        self.total_data = dataTensor
        self.transposed_data = self.total_data.T


class Tensorboard_node(Tensorboard):
    def __init__(self, dataTensor, path, file_name, config):
        super(Tensorboard_node, self).__init__()


class Tensorboard_elem(Tensorboard):
    def __init__(self, dataTensor, path, file_name, config):
        super(Tensorboard_elem, self).__init__()

    def time_write(self):
        # Gradient of node write in time
        # x: time
        # y: sum of grad (each node), norm of grad (each node), norm of grad (each layer)
        for t, data in enumerate(self.total_data):
            tmp_data = data.clone().detach()
            if t % 1000 == 0:
                print('\r {} line complete'.format(t), end='')
            for l, (num_w, num_b) in enumerate(zip(self.w_size_list, self.b_size_list)):
                # weight
                tmp_w = tmp_data[:num_w]
                tmp_data = tmp_data[num_w:]  # remove
                # self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm in layer(all elem)
                if self.NN_type_list[l] == 'cnn':
                    nodes_integrated_sum = dict()
                    nodes_integrated_norm = dict()
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        node_w = tmp_w[:(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                        nodes_integrated_sum['{}l_{}n'.format(
                            l, n)] = node_w.sum()
                        nodes_integrated_norm['{}l_{}n'.format(
                            l, n)] = node_w.norm(2)
                        # self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        # self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w = tmp_w[(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # 내용 제거

                elif self.NN_type_list[l] == 'fc':
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        node_w = tmp_w[:self.NN_size_list[l]]
                        nodes_integrated_sum['{}l_{}n'.format(
                            l, n)] = node_w.sum()
                        nodes_integrated_norm['{}l_{}n'.format(
                            l, n)] = node_w.norm(2)
                        # self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        # self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w = tmp_w[self.NN_size_list[l]:]  # 내용제거
                self.timeWriter.add_scalars(
                    '{}l/sum_of_grads', nodes_integrated_sum, t)
                self.timeWriter.add_scalars(
                    '{}l/l2_norm_of_grads', nodes_integrated_norm, t)

            # bias
            tmp_b = tmp_data[:num_b].detach().clone()
            tmp_data = tmp_data[num_b:]  # remove

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
                        '{}l/l2_norm_grad', node_w.norm(2), n)

            elif self.NN_type_list[l] == 'fc':
                for n in range(self.NN_size_list[l+1]):  # node 단위
                    node_w = tmp_w[:self.NN_size_list[l]]
                    tmp_w = tmp_w[self.NN_size_list[l]:]  # 내용제거
                    self.nodeWriter.add_scalar(
                        '{}l/l2_norm_grad', node_w.norm(2), n)

            # bias
            tmp_b = tmp_data[:num_b].detach().clone()
            tmp_data = tmp_data[num_b:]  # remove
