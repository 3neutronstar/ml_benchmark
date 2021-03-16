import matplotlib.pyplot as plt
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from Visualization.using_tensorboard import Tensorboard_elem, Tensorboard_node
from Visualization.using_pyplot import Pyplot_node


def using_tensorboard(fileTensor, config, path, file_name):
    print('Using tensorboard')
    epoch_rows = math.ceil(60000.0/float(config['batch_size']))
    config['epoch_rows'] = epoch_rows
    if file_name is None:
        file_name = 'grad'

    # if config['nn_type'] == 'lenet5':
    if True:
        logger = Tensorboard_elem(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            # logger.time_write()
            logger.time_write_elem()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()
    else:
        logger = Tensorboard_node(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()

    print('\n ==Visualization Complete==')


def using_plt(fileTensor, config, path,file_name):
    print('Using pyplot')
    logger = Pyplot_node(fileTensor, config, path,file_name)
    logger.time_write()
    logger.time_integrated_write()
    print('\n ==Visualization Complete==')
