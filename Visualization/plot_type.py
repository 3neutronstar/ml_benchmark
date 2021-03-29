import matplotlib.pyplot as plt
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from Visualization.using_tensorboard import Tensorboard_elem, Tensorboard_node,Tensorboard_node_big
from Visualization.using_pyplot import Pyplot_node


def using_tensorboard(fileTensor, config, path, file_name):
    print('Using tensorboard')
    epoch_rows = math.ceil(60000.0/float(config['batch_size']))
    config['epoch_rows'] = epoch_rows
    if file_name is None:
        file_name = 'grad'

    if config['nn_type'] == 'lenet5':
        logger = Tensorboard_node(fileTensor, path, file_name, config) #TODO REMOVE #Tensorboard_elem(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'time_elem_domain':
            logger.time_write_elem()
        elif config['visual_type']=='dist_domain':
            logger.dist_write()
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


    print('\n ==Visualization Complete==')


def using_plt(fileTensor, config, path,file_name):
    print('Using pyplot')
    logger = Pyplot_node(fileTensor, config, path,file_name)
    logger.time_write()
    logger.time_integrated_write()
    print('\n ==Visualization Complete==')
