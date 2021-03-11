import time
import torch
from torch.utils.tensorboard import SummaryWriter

class Tensorboard():
    def __init__(self,dataTensor,path,file_name,config):

        if config['nn_type']=='lenet5':
            from NeuralNet.lenet5 import w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list
        elif config['nn_type'][:3]=='vgg':
            from NeuralNet.vgg import get_nn_config
            w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list=get_nn_config(config['nn_type'])
        self.w_size_list=w_size_list
        self.b_size_list=b_size_list
        self.NN_size_list=NN_size_list
        self.NN_type_list=NN_type_list
        self.kernel_size_list=kernel_size_list
        if config['visual_type']=='node_domain':
            self.nodeWriter=SummaryWriter(log_dir='visualizing_data/{}/node_domain(x)'.format(file_name),)
        elif config['visual_type']=='time_domain':
            self.timeWriter=SummaryWriter(log_dir='visualizing_data/{}/time_domain(x)'.format(file_name))
        elif config['visual_type']=='node_domain_integrated':
            # node value integrated for each layer
            self.integratedNodeWriter=SummaryWriter(log_dir='visualizing_data/{}/node_domain(x)_Integrated'.format(file_name))
        self.total_data=dataTensor
        self.transposed_data=self.total_data.T
    
    def time_write(self):
        # Gradient of node write in time
        # x: time
        # y: sum of grad (each node), norm of grad (each node), norm of grad (each layer)
        for t,data in enumerate(self.total_data):
            tmp_data=data.clone().detach()
            if t%1000==0:
                print('\r {} line complete'.format(t),end='')
            for l,(num_w,num_b) in enumerate(zip(self.w_size_list,self.b_size_list)):
                #weight
                tmp_w=tmp_data[:num_w]
                tmp_data=tmp_data[num_w:]#remove
                self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm in layer(all elem)
                if self.NN_type_list[l]=='cnn':
                    for n in range(self.NN_size_list[l+1]):#node 단위
                        node_w=tmp_w[:(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                        self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w=tmp_w[(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]# 내용 제거

                elif self.NN_type_list[l]=='fc':
                    for n in range(self.NN_size_list[l+1]):#node 단위
                        node_w=tmp_w[:self.NN_size_list[l]]
                        self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                        self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w= tmp_w[self.NN_size_list[l]:] # 내용제거

            #bias
            tmp_b=tmp_data[:num_b].detach().clone()
            tmp_data=tmp_data[num_b:]#remove

    
    def node_write(self):
        sum_time=torch.sum(self.transposed_data,dim=0)
        for l,(num_w,num_b) in enumerate(zip(self.w_size_list,self.b_size_list)):
            tmp_w=tmp_data[:num_w]
            tmp_data=tmp_data[num_w:]#remove
            self.timeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm in layer(all elem)
            if self.NN_type_list[l]=='cnn':
                for n in range(self.NN_size_list[l+1]):#node 단위
                    node_w=tmp_w[:(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                    self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                    self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                    tmp_w=tmp_w[(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]# 내용 제거

            elif self.NN_type_list[l]=='fc':
                for n in range(self.NN_size_list[l+1]):#node 단위
                    node_w=tmp_w[:self.NN_size_list[l]]
                    self.timeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
                    self.timeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
                    tmp_w= tmp_w[self.NN_size_list[l]:] # 내용제거

            #bias
            tmp_b=tmp_data[:num_b].detach().clone()
            tmp_data=tmp_data[num_b:]#remove

    def layer_write(self):
        for l,(num_w,num_b) in enumerate(zip(self.w_size_list,self.b_size_list)):
            print('hi')
