import time
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

class Tensorboard():
    def __init__(self, dataTensor, path, file_name, configs):
        if configs['nn_type'] == 'lenet5':
            from NeuralNet.lenet5 import LeNet5
            model=LeNet5(configs)
            w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list=model.get_configs()
            self.kernel_size_list = kernel_size_list
        elif configs['nn_type'] == 'lenet300_100':
            from NeuralNet.lenet300_100 import LeNet_300_100
            model=LeNet_300_100(configs)
            w_size_list, b_size_list, NN_size_list, NN_type_list=model.get_configs()
        elif configs['nn_type'][:3] == 'vgg':
            from NeuralNet.vgg import VGG
            model=VGG(configs)
            w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list = model.get_configs(configs['nn_type'])
            self.kernel_size_list = kernel_size_list
        self.w_size_list = w_size_list
        self.b_size_list = b_size_list
        self.NN_size_list = NN_size_list
        self.NN_type_list = NN_type_list
        self.path=path
        if configs['visual_type'] == 'node_domain':
            self.nodeWriter = SummaryWriter(
                log_dir=os.path.join(path,'{}/node_info'.format(file_name)))
        if configs['visual_type'] == 'time_domain':
            self.timeWriter=list()
            self.timeWriter_cum=list()
            for l,_ in enumerate(b_size_list):
                self.timeWriter.append(SummaryWriter(log_dir=os.path.join(path,'time_info/{}/{}l'.format(file_name,l))))
                self.timeWriter_cum.append(SummaryWriter(log_dir=os.path.join(path,'time_info_cum/{}/{}l'.format(file_name,l))))
        if configs['visual_type'] == 'time_elem_domain':
            self.timeWriter=list()
            self.timeWriter_cum=list()
            for l,_ in enumerate(b_size_list):
                self.timeWriter.append(SummaryWriter(log_dir=os.path.join(path,'time_elem_info/{}/{}l'.format(file_name,l))))
                self.timeWriter_cum.append(SummaryWriter(log_dir=os.path.join(path,'time_elem_info_cum/{}/{}l'.format(file_name,l))))

        if configs['visual_type'] == 'node_domain_integrated':
            # node value integrated for each layer
            self.integratedNodeWriter = SummaryWriter(
                log_dir='visualizing_data/{}/node_xaxis_Integrated'.format(file_name))
        self.total_data = dataTensor
        self.transposed_data = self.total_data.T
        self.nodes_integrated = dict()
        self.node_elems_integrated=dict()# [l_n]=torch.tensor(elems,time)
        self.time_list = list()
        self.file_name=file_name
        self.info_type_list = [ 
                               'norm', 'norm_cum', ]#'var', 'var_cum','avg', 'avg_cum']


class Tensorboard_node(Tensorboard):  # norm avg기반
    def __init__(self, dataTensor, path, file_name, configs):
        super(Tensorboard_node, self).__init__(
            dataTensor, path, file_name, configs)
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
                for n in range(num_w):  # node 단위
                    if t == 0:
                        self.nodes_integrated['avg_{}l_{}n'.format(
                            l, n)] = list()
                        self.nodes_integrated['norm_{}l_{}n'.format(
                            l, n)] = list()
                        self.nodes_integrated['var_{}l_{}n'.format(
                            l, n)] = list()

                    self.nodes_integrated['avg_{}l_{}n'.format(
                        l, n)].append(node_w[n][0])
                    self.nodes_integrated['norm_{}l_{}n'.format(
                        l, n)].append(node_w[n][1])
                    self.nodes_integrated['var_{}l_{}n'.format(
                        l, n)].append(node_w[n][2])

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
            self.timeWriter_cum[layer].add_scalar(
                '{}/{}l_{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)
            
        else:
            self.timeWriter[layer].add_scalar(
                '{}/{}l_{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)
        #self.nodes_integrated.pop('{}_{}l_{}n'.format(info_type,layer, node))

    def time_write(self):
        for type_info in self.info_type_list:
            for l_idx,num_node in enumerate(self.b_size_list):
                if 'cum' in type_info:
                    for t in self.time_list:
                        layer_dict=dict()
                        for n_idx in range(num_node):
                            layer_dict['{}n'.format(n_idx)]=self.nodes_integrated['{}_{}l_{}n'.format(type_info,l_idx,n_idx)][t]
                        self.timeWriter_cum[l_idx].add_scalars(type_info,layer_dict,t)
                    self.timeWriter_cum[l_idx].flush()
                else:
                    for t in self.time_list:
                        layer_dict=dict()
                        for n_idx in range(num_node):
                            layer_dict['{}n'.format(n_idx)]=self.nodes_integrated['{}_{}l_{}n'.format(type_info,l_idx,n_idx)][t]
                        self.timeWriter[l_idx].add_scalars(type_info,layer_dict,t)
                    self.timeWriter[l_idx].flush()
                print('\r{}_{}l Complete====='.format(type_info,l_idx),end='')
    
    def time_write_integrated_(self, layer, node, info_type,t):
        #TODO
        # plt.clf()
        # plt.plot(self.time_list, self.nodes_integrated['{}_{}l_{}n'.format(
        #     info_type,layer, node)])
        if 'cum' in info_type:
            self.timeWriter_cum.add_scalars(
                '{}/{}l/{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)
            
        else:
            self.timeWriter.add_scalars(
                '{}/{}l/{}n'.format(info_type,layer, node),self.nodes_integrated['{}_{}l_{}n'.format(info_type,layer, node)][t],t)
        #self.nodes_integrated.pop('{}_{}l_{}n'.format(info_type,layer, node))

    def time_write_integrated(self):
        for info_type in self.info_type_list:
            for l, num_node in enumerate(self.b_size_list):
                for n in range(num_node):
                    print("\rinfo_type: {}_{}l_{}n==========".format(info_type,l,n),end='')
                    for t in self.time_list:
                        self.time_write_(l, n, info_type,t)
                    self.nodes_integrated.pop('{}_{}l_{}n'.format(info_type,l, n))
            self.timeWriter.flush()
            self.timeWriter_cum.flush()


    def node_write(self):
        # print(self.total_data.size())
        sum_data = torch.mean(self.total_data, dim=0).squeeze(0)
        # print(sum_data.size())
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
    def __init__(self, dataTensor, path, file_name, configs):
        super(Tensorboard_elem, self).__init__(
            dataTensor, path, file_name, configs)
        self.num_elem_list=list()
        if os.path.exists(os.path.join(self.path,'time_elem_info')) == False:
            os.mkdir(os.path.join(self.path,'time_elem_info'))
        if os.path.exists(os.path.join(self.path,'time_elem_info_cum')) == False:
            os.mkdir(os.path.join(self.path,'time_elem_info_cum'))
        if os.path.exists(os.path.join(self.path,'time_elem_info',self.file_name)) == False:
            os.mkdir(os.path.join(self.path,'time_elem_info',self.file_name))
        if os.path.exists(os.path.join(self.path,'time_elem_info_cum',self.file_name)) == False:
            os.mkdir(os.path.join(self.path,'time_elem_info_cum',self.file_name))

        # Gradient of node write in time
        # x: time
        # y: sum of grad (each node), norm of grad (each node), norm of grad (each layer)

    def time_write_(self):
        for t, data in enumerate(self.total_data):
            self.time_list.append(t)
            tmp_data = data.clone().detach()
            if t % 1000 == 0:
                print('\r {} line complete'.format(t), end='')
            for l, (num_w, num_b) in enumerate(zip(self.w_size_list, self.b_size_list)):
                # self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(),t)#norm in layer(all elem)
                if self.NN_type_list[l] == 'cnn':
                    # weight
                    tmp_w = tmp_data[:num_b*(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]].detach().clone()
                    tmp_data = tmp_data[num_b*(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # remove
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        if t == 0:
                            self.nodes_integrated['avg_{}l_{}n'.format(
                                l, n)] = list()
                            self.nodes_integrated['norm_{}l_{}n'.format(
                                l, n)] = list()
                            self.nodes_integrated['var_{}l_{}n'.format(
                                l, n)] = list()
                            self.num_elem_list.append((self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l])
                        node_w = tmp_w[:(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]].detach().clone()
                        self.nodes_integrated['avg_{}l_{}n'.format(
                            l, n)].append(node_w.mean())
                        # print(node_w,t)
                        self.nodes_integrated['norm_{}l_{}n'.format(
                            l, n)].append(node_w.norm())
                        self.nodes_integrated['var_{}l_{}n'.format(
                            l, n)].append(node_w.var())
                        tmp_w = tmp_w[(
                            self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # 내용 제거

                elif self.NN_type_list[l] == 'fc':
                    # weight
                    tmp_w = tmp_data[:num_b*self.NN_size_list[l]].detach().clone()
                    tmp_data = tmp_data[num_b*self.NN_size_list[l]:]  # remove
                    for n in range(self.NN_size_list[l+1]):  # node 단위
                        if t == 0:
                            self.nodes_integrated['avg_{}l_{}n'.format(
                                l, n)] = list()
                            self.nodes_integrated['norm_{}l_{}n'.format(
                                l, n)] = list()
                            self.nodes_integrated['var_{}l_{}n'.format(
                                l, n)] = list()
                            self.num_elem_list.append(self.NN_size_list[l]*self.NN_size_list[l+1])
                        node_w = tmp_w[:self.NN_size_list[l]].detach().clone()
                        self.nodes_integrated['avg_{}l_{}n'.format(
                            l, n)].append( node_w.mean())
                        self.nodes_integrated['norm_{}l_{}n'.format(
                            l, n)].append(node_w.norm())
                        self.nodes_integrated['var_{}l_{}n'.format(
                            l, n)].append (node_w.var())
                        tmp_w = tmp_w[self.NN_size_list[l]:]  # 내용제거
                # # bias
                # node_b = tmp_data[:num_b].detach().clone()
                # tmp_data = tmp_data[num_b:]  # remove
        for type_info in self.info_type_list:
            for l,num_node in enumerate(self.b_size_list):
                for n in range(num_node):
                    self.nodes_integrated['{}_cum_{}l_{}n'.format(type_info,l,n)]=torch.cumsum(torch.tensor(self.nodes_integrated['{}_{}l_{}n'.format(type_info,l, n)]), dim=0).clone().tolist()

        print("\n Reading Process Complete")

    
    def time_write(self):
        self.time_write_()
        for type_info in self.info_type_list:
            for l_idx,num_node in enumerate(self.b_size_list):
                if 'cum' in type_info:
                    for t in self.time_list:
                        layer_dict=dict()
                        for n_idx in range(num_node):
                            layer_dict['{}n'.format(n_idx)]=self.nodes_integrated['{}_{}l_{}n'.format(type_info,l_idx,n_idx)][t]
                        self.timeWriter_cum[l_idx].add_scalars(type_info,layer_dict,t)
                    self.timeWriter_cum[l_idx].flush()
                else:
                    for t in self.time_list:
                        layer_dict=dict()
                        for n_idx in range(num_node):
                            layer_dict['{}n'.format(n_idx)]=self.nodes_integrated['{}_{}l_{}n'.format(type_info,l_idx,n_idx)][t]
                        self.timeWriter[l_idx].add_scalars(type_info,layer_dict,t)
                    self.timeWriter[l_idx].flush()
                print('\r{}_{}l Complete====='.format(type_info,l_idx),end='')
        del self.nodes_integrated
    
    def time_write_elem_(self):
        data_dict=dict() # layer name:
        for t in range(self.total_data.size()[0]):
            self.time_list.append(t)

        tmp_data = self.total_data.clone().detach()
        for l, (num_w, num_b) in enumerate(zip(self.w_size_list, self.b_size_list)):
            # self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(),t)#norm in layer(all elem)
            if self.NN_type_list[l] == 'cnn':
                # weight
                tmp_w = tmp_data[:,:num_b*(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]].detach().clone()
                tmp_data = tmp_data[:,num_b*(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]  # remove
                for elem_idx,w in enumerate(tmp_w.T):
                    if t==0:
                        data_dict['{}l_{}e'.format(l,elem_idx)]=list()
                    data_dict['{}l_{}e'.format(l,elem_idx)]=w
                    data_dict['cum_{}l_{}e'.format(l,elem_idx)]=torch.cumsum(w,dim=0)
                self.num_elem_list.append((self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l])

            elif self.NN_type_list[l] == 'fc':
                # weight
                
                tmp_w = tmp_data[:,:num_b*self.NN_size_list[l]].detach().clone()
                tmp_data = tmp_data[:,num_b*self.NN_size_list[l]:]  # remove
                for elem_idx,w in enumerate(tmp_w.T):
                    if t==0:
                        data_dict['{}l_{}e'.format(l,elem_idx)]=list()
                    data_dict['{}l_{}e'.format(l,elem_idx)]=w
                    data_dict['cum_{}l_{}e'.format(l,elem_idx)]=torch.cumsum(w,dim=0)
                self.num_elem_list.append(self.NN_size_list[l]*self.NN_size_list[l+1])
            print('{} layer done, {}'.format(l,tmp_w.T.size()))

        return data_dict

    def time_write_elem(self):
        
        data_dict=self.time_write_elem_()
        for l_idx,num_node in enumerate(self.b_size_list):
            if l_idx==2:
                for n in range(num_node):
                    if self.NN_type_list[l_idx] == 'cnn':
                        node_size=self.kernel_size_list[l_idx][0]*self.kernel_size_list[l_idx][1]*self.NN_size_list[l_idx]
                        for e in range(node_size):
                            for t in self.time_list:
                                self.timeWriter[l_idx].add_scalar('{}n/{}e'.format(n,e),data_dict['{}l_{}e'.format(l_idx,e+node_size*n)][t],t)
                                self.timeWriter_cum[l_idx].add_scalar('{}n/{}e'.format(n,e),data_dict['cum_{}l_{}e'.format(l_idx,e+node_size*n)][t],t)
                            self.timeWriter[l_idx].flush()
                            self.timeWriter_cum[l_idx].flush()
                            print('\r {}l_{}n_{}e complete'.format(l_idx,n,e),end='')
                    if self.NN_type_list[l_idx] == 'fc':
                        node_size=self.NN_size_list[l_idx]
                        for e in range(node_size):
                            for t in self.time_list:
                                self.timeWriter[l_idx].add_scalar('{}n/cum_{}e'.format(n,e),data_dict['{}l_{}e'.format(l_idx,e+node_size*n)][t],t)
                                self.timeWriter_cum[l_idx].add_scalar('{}n/cum_{}e'.format(n,e),data_dict['cum_{}l_{}e'.format(l_idx,e+node_size*n)][t],t)
                            self.timeWriter[l_idx].flush()
                            self.timeWriter_cum[l_idx].flush()
                            print('\r {}l_{}n_{}e complete'.format(l_idx,n,e),end='')


                    # t_data=dict()
                    # t_data_cum=dict()
                    # print('\r {} Layer {:7.0f}Time Complete'.format(l_idx,t),end='')
                    # for e_idx in range(self.num_elem_list[l_idx]*num_node):
                    #     t_data['{}l_{}e'.format(l_idx,e_idx)]=data_dict['{}l_{}e'.format(l_idx,e_idx)][t]
                    # self.timeWriter[l_idx].add_scalars('elem_grad',t_data,t)
                    # self.timeWriter[l_idx].flush()
                    # del t_data

                    # for e_idx in range(self.num_elem_list[l_idx]*num_node):
                    #     t_data_cum['{}l_{}e'.format(l_idx,e_idx)]=data_dict['cum_{}l_{}e'.format(l_idx,e_idx)][t]
                    # self.timeWriter_cum[l_idx].add_scalars('cum_elem_grad',t_data,t)
                    # self.timeWriter_cum[l_idx].flush()
                    # del t_data_cum

                    


        # for l_idx,num_node in enumerate(self.b_size_list):
        #     cum_save_path=os.path.join(self.path,'time_elem_info_cum',self.file_name,'{}l'.format(l_idx))
        #     save_path=os.path.join(self.path,'time_elem_info',self.file_name,'{}l'.format(l_idx))
        #     if os.path.exists(os.path.join(self.path,'time_elem_info_cum',self.file_name,'{}l'.format(l_idx))) == False:
        #         os.mkdir(os.path.join(self.path,'time_elem_info_cum',self.file_name,'{}l'.format(l_idx)))
        #     if os.path.exists(os.path.join(self.path,'time_elem_info',self.file_name,'{}l'.format(l_idx))) == False:
        #         os.mkdir(os.path.join(self.path,'time_elem_info',self.file_name,'{}l'.format(l_idx)))
        #     for e_idx in range(self.num_elem_list[l_idx]*num_node):
        #         print('\r {} Layer {:4.0f}Elem Complete'.format(l_idx,e_idx),end='')
        #         plt.clf()
        #         plt.plot(self.time_list,data_dict['{}l_{}e'.format(l_idx,e_idx)])
        #         plt.ylabel('grad_elem')
        #         plt.xlabel('time')
        #         plt.savefig(os.path.join(save_path,'{}l_{}e.png'.format(l_idx,e_idx)),dpi=100)

        #         plt.clf()
        #         plt.plot(self.time_list,torch.cumsum(data_dict['{}l_{}e'.format(l_idx,e_idx)],dim=0))
        #         plt.ylabel('grad_elem')
        #         plt.xlabel('time')
        #         plt.savefig(os.path.join(cum_save_path,'{}l_{}e.png'.format(l_idx,e_idx)),dpi=100)
