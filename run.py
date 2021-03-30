from Visualization.create_cam import create_cam
import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import random
import numpy as np
from utils import load_params, save_params

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or visual, cam, prune,visual_prune, train_prune, test, extract_npy')
    
    #TRAIN SECTION
    parser.add_argument(
        '--seed', type=int, default=1,
        help='fix random seed')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='set mini-batch size')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='set mini-batch size')
    parser.add_argument(
        '--nn_type', type=str, default='lenet5',
        help='choose NeuralNetwork type')
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='choose NeuralNetwork')
    parser.add_argument(
        '--colab', type=bool, default=False,
        help='if you are in colab use it')
    parser.add_argument(
        '--num_workers', type=int, default=3,
        help='number of process you have')
    parser.add_argument(
        '--log', type=str, default='true',
        help='generate log')
    # save grad
    parser.add_argument(
        '--grad_save', type=str, default='true',
        help='generate grad_save')
    # prune threshold
    parser.add_argument(
        '--threshold', type=int, default=100,
        help='set prune threshold by cum of norm in elems')
    # prune threshold
    parser.add_argument(
        '--grad_off_epoch', type=int, default=5,
        help='set gradient off and prune start epoch')


    #TRAIN OPTION BY NN
    nn_type = parser.parse_known_args(args)[0].nn_type.lower()
    if nn_type == 'lenet5':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9
    elif nn_type == 'vgg16':
        dataset = 'cifar10'
        epochs = 300
        lr=1e-2
        momentum=0.9
    elif nn_type=='lenet300_100':
        dataset = 'mnist'
        epochs = 60
        lr=1e-2
        momentum=0.9

    parser.add_argument(
        '--lr', type=float, default=lr,
        help='set learning rate')
    parser.add_argument(
        '--momentum', type=float, default=momentum,
        help='set learning rate')
    parser.add_argument(
        '--epochs', type=int, default=epochs,
        help='run epochs')
    parser.add_argument(
        '--dataset', type=str, default=dataset,
        help='choose dataset, if nn==lenet5,mnist elif nn==vgg16,cifar10')
    
    # VISUAL and CAM SECTION
    parser.add_argument(
        '--file_name', type=str, default=None,
        help='grad_data/grad_[].log for VISUAL and grad_[].pt file load for CAM')
    parser.add_argument(
        '--visual_type', type=str, default='time_domain',
        help='visualization domain decision [time,node,node_integrated]')
    parser.add_argument(
        '--num_result', type=int, default=1,
        help='grad_data/grad_[].log file load')
    
    parser.add_argument(
        '--test_seed', type=int, default=1,
        help='remove')#TODO
    
    
    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    train_mode_list=['train','train_prune']
    if flags.file_name is None and flags.mode in train_mode_list:
        time_data = time.strftime(
            '%m-%d_%H-%M-%S', time.localtime(time.time()))
        print(time_data)
        if os.path.exists(os.path.dirname(os.path.join(os.path.abspath(__file__),'grad_data'))) == False:
            os.mkdir(os.path.dirname(os.path.join(os.path.abspath(__file__),'grad_data')))
    elif flags.file_name is not None and flags.mode not in train_mode_list:  # load param when not training
        time_data = flags.file_name
        file_name = flags.file_name
    else:
        file_name = None  # no file name just read from grad.csv, .npy and .pt
    
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and flags.device == 'gpu' else "cpu")
    # Random Seed 설정
    random_seed = flags.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''
    Basic Setting
    '''
    configs = {'device': str(device),
               'seed': random_seed,
               'epochs': flags.epochs,
               'lr': flags.lr,
               'batch_size': flags.batch_size,
               'dataset': flags.dataset.lower(),
               'nn_type': flags.nn_type.lower(),
               'colab': flags.colab,
               'log_extraction': flags.log.lower(),
               'num_workers': flags.num_workers,
               'visual_type':flags.visual_type,
               'mode':flags.mode,
               'patience':flags.patience,
               'momentum':flags.momentum,
               'grad_save':flags.grad_save.lower(),
               'threshold':flags.threshold,
               'grad_off_epoch':flags.grad_off_epoch,
               'test_seed':flags.test_seed,
               }
    print("SEED:",flags.test_seed)
    # print(flags.log)
    if configs['log_extraction'] == 'true' and configs['mode'] in train_mode_list:
        save_params(configs, time_data)
        print("Using device: {}, Mode:{}, Type:{}".format(device,flags.mode,flags.nn_type))
        sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'grad_data','log_{}.txt'.format(time_data)),'w')
    else:
        if flags.file_name is not None:
            CALL_CONFIG = load_params(configs, file_name)
            CALL_CONFIG['mode']=configs['mode']
            if configs['mode']=='visual':
                CALL_CONFIG['visual_type']=configs['visual_type']
                print(configs['visual_type'])
            configs=CALL_CONFIG
            print("Mode:{}, Type:{}".format(configs['mode'],configs['nn_type']))
        
    if flags.mode == 'visual':
        from visualization import visualization
        configs = visualization(configs, file_name)
    else:
        if configs['nn_type'] == 'lenet5':
            from NeuralNet.lenet5 import LeNet5
            model = LeNet5(configs).to(configs['device'])
        if configs['nn_type'][:3] == 'vgg':
            from NeuralNet.vgg import VGG
            model = VGG(configs).to(configs['device'])
            # print(model)
        if configs['nn_type']=='lenet300_100':
            from NeuralNet.lenet300_100 import LeNet_300_100
            model = LeNet_300_100(configs).to(configs['device'])
    # time_data
    # sys.

    if flags.mode == 'train' or flags.mode=='train_prune':
        from train import Learner
        learner=Learner(model,time_data,configs)
        configs=learner.run()
    elif flags.mode=='extract_npy':
        from train import Learner
        learner=Learner(model,time_data,configs)
        configs=learner.save_grad(configs['end_epoch'])


    if flags.mode.lower() =='cam':
        configs['batch_size']=1 # 1장씩 extracting
        file_path=os.path.dirname(os.path.abspath(__file__))
        create_cam(model,file_path,file_name,flags.num_result,configs)
    if flags.mode.lower()=='visual_prune':
        from Visualization.visual_prune import visual_prune
        configs['batch_size']=1 # 1장씩 extracting
        file_path=os.path.dirname(os.path.abspath(__file__))
        visual_prune(model,file_path,file_name,configs)
    
    print("End the process")
    if flags.log.lower()=='true':
        sys.stdout.close()

if __name__ == '__main__':
    main(sys.argv[1:])
