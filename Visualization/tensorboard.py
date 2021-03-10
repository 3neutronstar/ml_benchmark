import torch
from torch.utils.tensorboard import SummaryWriter

class Tensorboard():
    def __init__(self,csvTensor,path,config):

        if config['nn_type']=='lenet5':
            from NeuralNet.lenet5 import w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list
        elif config['nn_type']=='vgg16':
            from NeuralNet.vgg16 import w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list
        self.w_size_list=w_size_list
        self.b_size_list=b_size_list
        self.NN_size_list=NN_size_list
        self.NN_type_list=NN_type_list
        self.kernel_size_list=kernel_size_list
        if config['visual_type']=='elem':
            self.elemWriter=SummaryWriter(log_dir='visualizing_data/elem_info',)
        elif config['visual_type']=='node':
            self.nodeWriter=SummaryWriter(log_dir='visualizing_data/node_info')
        
        total_data_list=list()
        for t,line in enumerate(csvTensor):
            total_data_list.append(torch.tensor(line).clone().detach())
            if t%1000==0:
                print('\r {} line complete'.format(t),end='')
        
        self.total_data=torch.cat(total_data_list,dim=0)
        tmp_data=self.total_data.detach().clone()
    
    def node_write(self):
        for t,data in enumerate(self.total_data):
            if t%1000==0:
                print('\r {} line complete'.format(t),end='')
            for l,(num_w,num_b) in enumerate(zip(self.w_size_list,self.b_size_list)):
                #weight
                tmp_w=tmp_data[:num_w]
                tmp_data=tmp_data[num_w:]#remove
                if self.NN_type_list[l]=='cnn':
                    for n in range(self.NN_size_list[l+1]):#node 단위
                        node_w=tmp_w[:(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]]
                        self.nodeWriter.add_scalar('{}l_{}n/sum_grad'.format(l,n),node_w.sum(),t)#합
                        self.nodeWriter.add_scalar('{}l_{}n/norm_grad'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w=tmp_w[(self.kernel_size_list[l][0]*self.kernel_size_list[l][1])*self.NN_size_list[l]:]# 내용 제거

                elif self.NN_type_list[l]=='fc':
                    for n in range(self.NN_size_list[l+1]):#node 단위
                        node_w=tmp_w[:self.NN_size_list[l]]
                        self.nodeWriter.add_scalar('{}l_{}n/sum_grad'.format(l,n),node_w.sum(),t)#합
                        self.nodeWriter.add_scalar('{}l_{}n/norm_grad'.format(l,n),node_w.norm(2),t)#norm
                        tmp_w= tmp_w[self.NN_size_list[l]:] # 내용제거

            #bias
            tmp_b=tmp_data[:num_b].detach().clone()
            tmp_data=tmp_data[num_b:]#remove