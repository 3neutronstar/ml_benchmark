import matplotlib.pyplot as plt
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from Visualization.tensorboard import Tensorboard_elem, Tensorboard_node


def using_tensorboard(fileTensor, config, path, file_name):
    print('Using tensorboard')
    epoch_rows = math.ceil(60000.0/float(config['batch_size']))
    config['epoch_rows'] = epoch_rows
    if file_name is None:
        file_name = 'grad'

    # if config['nn_type'] == 'lenet5':
    if False:
        logger = Tensorboard_elem(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()
    else:
        logger = Tensorboard_node(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()

    # if config['nn_type']=='lenet5':
    #     from NeuralNet.lenet5 import w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list
    # if False:
    #     elemWriter=SummaryWriter(log_dir='visualizing_data/elem_info',)
    # else:
    #     nodeWriter=SummaryWriter(log_dir='visualizing_data/node_info/{}'.format(file_name))
    #     total_data_list=list()
    #     for t,line in enumerate(fileTensor):
    #         total_data_list.append(torch.tensor(line).clone().detach())

    #         if t%1000==0:
    #             print('\r {} line complete'.format(t),end='')
    #         for l,(num_w,num_b) in enumerate(zip(w_size_list,b_size_list)):# 라인 내의 모든 param처리
    #             tmp_w=torch.tensor(line[:num_w],dtype=torch.float).clone().detach()#layer 단위
    #             line=line[num_w:]
    #             nodeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm
    #             if NN_type_list[l]=='cnn':
    #                 n=0
    #                 for n in range(NN_size_list[l+1]):#node 단위
    #                     node_w=tmp_w[:(kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]]
    #                     # print(tmp_w.size(),node_w.size(),NN_type_list[l],n," ",l," ",t)
    #                     nodeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
    #                     nodeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
    #                     tmp_w=tmp_w[(kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]:]# 내용 제거

    #             elif NN_type_list[l]=='fc':
    #                 n=0
    #                 for n in range(NN_size_list[l+1]):#node 단위
    #                     node_w=tmp_w[:NN_size_list[l]]
    #                     nodeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
    #                     nodeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
    #                     tmp_w= tmp_w[NN_size_list[l]:] # 내용제거

    #             tmp_b=torch.tensor(line[:num_b],dtype=torch.float).clone().detach()
    #             line=line[num_b:]#내용제거
    #             nodeWriter.flush()

    #     # 시간과 상관없는 분석을 위한 cat
    #     total_data=torch.cat(total_data_list,dim=0)
    #     nodeWriter.close()
    print('\n ==Visualization Complete==')


def using_plt(fileTensor, config, path):
    print('Using plt')
    NUM_ROWS = config['epochs']*math.ceil(60000.0/float(config['batch_size']))
    if config['nn_type'] == 'lenet5':
        from NeuralNet.lenet5 import w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list
    elif config['nn_type'][:3] == 'vgg':
        from NeuralNet.vgg import get_nn_config
        w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list = get_nn_config(
            config['nn_type'])

    nodes_integrated_avg_cum = dict()
    nodes_integrated_norm_cum = dict()
    nodes_integrated_var_cum = dict()
    nodes_integrated_avg = dict()
    nodes_integrated_norm = dict()
    nodes_integrated_var = dict()
    total_data=fileTensor.clone()
    time_list=list()
    for t, data in enumerate(total_data):
        tmp_data = data.detach().clone()
        time_list.append(t)
        if t % 100 == 0:
            print('\r {} line complete'.format(t), end='')
        for l, num_w in enumerate(b_size_list):  # b인 이유: node관찰이므로
            # weight
            node_w = tmp_data[:num_w].detach().clone()
            tmp_data = tmp_data[num_w:]
            for n, node_info in enumerate(node_w):  # node 단위
                if t==0:
                    nodes_integrated_avg['{}l_{}n'.format(l,n)]=list()
                    nodes_integrated_norm['{}l_{}n'.format(l,n)]=list()
                    nodes_integrated_var['{}l_{}n'.format(l,n)]=list()

                nodes_integrated_avg['{}l_{}n'.format(
                    l, n)].append(node_info[0])
                nodes_integrated_norm['{}l_{}n'.format(
                    l, n)].append(node_info[1])
                nodes_integrated_var['{}l_{}n'.format(
                    l, n)].append(node_info[2])
    
    for l,num_node in enumerate(b_size_list):
        for n in range(num_node):
            nodes_integrated_avg_cum=torch.cumsum(torch.tensor(nodes_integrated_avg['{}l_{}n'.format(l, n)]),dim=0)
            plt.clf()
            plt.plot([t,torch.tensor(nodes_integrated_avg['{}l_{}n'.format(l, n)]).tolist()])
            plt.xlabel('iter')
            plt.ylabel('avg of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_avg.png'),dpi=100)
            plt.clf()
            plt.plot([t,torch.tensor(nodes_integrated_avg_cum['{}l_{}n'.format(l, n)]).tolist()])
            plt.xlabel('iter')
            plt.ylabel('avg cum of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_avg_cum.png'),dpi=100)

            plt.clf()
            plt.plot([t,torch.tensor(nodes_integrated_var['{}l_{}n'.format(l, n)]).tolist()])
            plt.xlabel('iter')
            plt.ylabel('var of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_var.png'),dpi=100)
            plt.clf()
            plt.plot((t,nodes_integrated_var_cum['{}l_{}n'.format(l, n)].tolist()))
            plt.xlabel('iter')
            plt.ylabel('var cum of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_var_cum.png'),dpi=100)

            plt.clf()
            plt.plot((t,nodes_integrated_norm['{}l_{}n'.format(l, n)].tolist()))
            plt.xlabel('iter')
            plt.ylabel('norm of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_norm.png'),dpi=100)
            plt.clf()
            plt.plot((t,nodes_integrated_norm_cum['{}l_{}n'.format(l, n)].tolist()))
            plt.xlabel('iter')
            plt.ylabel('norm cum of grad in node')
            plt.savefig(os.path.join(path,'visualizing_data','node_info','{}l_{}n_norm_cum.png'),dpi=100)


