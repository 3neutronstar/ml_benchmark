import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import random
import numpy as np
from utils import load_params,save_params

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or test, simulate, "train_old" is the old version to train')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='fix random seed')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='set mini-batch size')
    parser.add_argument(
        '--epochs', type=int, default=60,
        help='run epochs')
    parser.add_argument(
        '--lr', type=float, default=1e-1,
        help='set learning rate')
    parser.add_argument(
        '--lr_decaying_period', type=int, default=15,
        help='set learning rate')
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        help='choose dataset')
    parser.add_argument(
        '--nn_type', type=str, default='lenet5',
        help='choose NeuralNetwork type')
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='choose NeuralNetwork')
    parser.add_argument(
        '--file_name', type=str, default=None,
        help='grad_data/[].csv file load')
    parser.add_argument(
        '--colab',type=bool,default=False,
        help='if you are in colab use it')

    return parser.parse_known_args(args)[0]

def main(args):
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    flags= parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and flags.device == 'gpu' else "cpu")
    print("Using device: {}".format(device))
    # Random Seed 설정
    random_seed = flags.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    


    '''
    Basic Setting
    '''
    configs={'device':str(device),
    'seed':random_seed,
    'epochs':flags.epochs,
    'lr':flags.lr,
    'lr_decaying_period':flags.lr_decaying_period,
    'batch_size':flags.batch_size,
    'dataset':flags.dataset.lower(),
    'nn_type':flags.nn_type.lower(),
    'colab':flags.colab,
    }


    if flags.mode=='train':
        from train import extract_data
        configs=extract_data(configs)
        save_params(configs,time_data)
    if flags.mode=='visual':
        from visualization import visualization
        configs=visualization(configs)
    
    print("End the process")
    





if __name__ == '__main__':
    main(sys.argv[1:])