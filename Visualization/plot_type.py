import matplotlib.pyplot as plt
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from Visualization.using_tensorboard import Tensorboard_elem, Tensorboard_node,Tensorboard_node_big
from Visualization.using_pyplot import Pyplot_node

def print_all_exp(fileTensor,path,file_name,config):
    dataFile = open(os.path.join(path, 'exp.txt'), mode='a')
    from NeuralNet.lenet5 import LeNet5
    model=LeNet5(config)
    w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list,node_size_list=model.get_configs()
    time_list=[]
    nodes_integrated=dict()
    for t, data in enumerate(fileTensor):
        time_list.append(t)
        tmp_data = data.detach().clone()
        if t % 100 == 0:
            print('\r {} line complete'.format(t), end='')
        for l, num_w in enumerate(node_size_list):  # b인 이유: node관찰이므로
            # weight
            node_w = tmp_data[:num_w].detach().clone()
            tmp_data = tmp_data[num_w:]
            for n in range(num_w):  # node 단위
                if t == 0:
                    nodes_integrated['norm_{}l_{}n'.format(
                        l, n)] = list()
                nodes_integrated['norm_{}l_{}n'.format(
                    l, n)].append(node_w[n][1])

    for l, num_w in enumerate(node_size_list):  # b인 이유: node관찰이므로
        if l==2:
            for n in range(num_w):  # node 단위
                write_str=file_name+"_{:>3}n: ".format(n)+str(float(torch.tensor(nodes_integrated['norm_{}l_{}n'.format(l,n)]).mean()))+" 468 "+str(float(nodes_integrated['norm_{}l_{}n'.format(l,n)][468]))+" 936 "+str(float(nodes_integrated['norm_{}l_{}n'.format(l,n)][936]))+"\n"
                dataFile.write(write_str)
                dataFile.flush()
    dataFile.write(" ")
    dataFile.close()


def using_tensorboard(config, path, file_name):
    print('Using tensorboard')
    epoch_rows = math.ceil(60000.0/float(config['batch_size']))
    config['epoch_rows'] = epoch_rows
    if file_name is None:
        file_name = 'grad'
    if config['visual_type']=='expectation':#TEST
        from visualization import load_npy
        dataTensor = load_npy(path, file_name)
        print_all_exp(dataTensor,path,file_name,config)#expectation of norm print #TODO REmove
    else:
        from visualization import load_npy
        fileTensor = load_npy(path, file_name)

        if config['nn_type'] == 'lenet5':
            logger = Tensorboard_node(fileTensor,path, file_name, config) #TODO REMOVE #Tensorboard_elem(fileTensor, path, file_name, config)
            if config['visual_type'] == 'time_domain':
                logger.time_write()
            elif config['visual_type'] == 'time_elem_domain':
                logger.time_write_elem()
            elif config['visual_type']=='dist_domain':
                logger.dist_write()
            elif config['visual_type'] == 'node_domain':
                logger.node_write()
            elif config['visual_type'] == 'node_domain_time': #TODO only in Tensorboard_node
                logger.node_write_time()
        elif config['nn_type'] == 'lenet300_100':
            logger = Tensorboard_node(fileTensor, path, file_name, config)
            if config['visual_type'] == 'time_domain':
                logger.time_write()
            elif config['visual_type'] == 'node_domain':
                logger.node_write()
            elif config['visual_type']=='dist_domain':
                logger.dist_write()
        else:
            logger = Tensorboard_node_big(fileTensor, path, file_name, config)
            if config['visual_type'] == 'time_domain':
                logger.time_write()
            elif config['visual_type'] == 'node_domain':
                logger.node_write()
            elif config['visual_type']=='dist_domain':
                logger.dist_write()
            elif config['visual_type'] == 'node_domain_time':
                logger.node_write_time()


    print('\n ==Visualization Complete==')


def using_plt(fileTensor, config, path,file_name):
    print('Using pyplot')
    logger = Pyplot_node(fileTensor, config, path,file_name)
    logger.time_write()
    logger.time_integrated_write()
    print('\n ==Visualization Complete==')
