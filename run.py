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
TRAIN_MODE=['train','train_weight_prune', 'train_grad_visual', 'train_lrp','train_mtl','train_mtl_v2','train_mtl_v4']
def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='visual[need saved grad], or test, load, {}'.format(TRAIN_MODE))
    
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
    parser.add_argument(
        '--earlystop', type=bool, default=False,
        help='earlystopping')
    # save grad
    parser.add_argument(
        '--grad_save', type=bool, default=False,
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
    from NeuralNet.baseNet import get_hyperparams
    dataset,epochs,lr,momentum=get_hyperparams(nn_type)

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
        '--start_epoch', type=int, default=1,help='for load model'
    )
    parser.add_argument(
        '--dataset', type=str, default=dataset,
        help='choose dataset, if nn==lenet5,mnist elif nn==vgg16,cifar10')
    
    # VISUAL and CAM SECTION
    parser.add_argument(
        '--file_name', type=str, default=None,
        help='grad_data/grad_[].log for VISUAL and grad_[].pt file load for CAM')
    parser.add_argument(
        '--visual_type', type=str, default='time_domain',
        help='visualization domain decision [time,node]')
    parser.add_argument(
        '--num_result', type=int, default=1,
        help='grad_data/grad_[].log file load')
        
    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    train_mode_list=TRAIN_MODE
    if flags.file_name is None and flags.mode in train_mode_list:
        time_data = time.strftime(
            '%m-%d_%H-%M-%S', time.localtime(time.time()))
        print(time_data)
        if os.path.exists(os.path.dirname(os.path.join(os.path.abspath(__file__),'grad_data'))) == False and flags.log =='true':
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
               'start_epoch':flags.start_epoch,
               'lr': flags.lr,
               'batch_size': flags.batch_size,
               'dataset': flags.dataset.lower(),
               'nn_type': flags.nn_type.lower(),
               'colab': flags.colab,
               'log_extraction': flags.log.lower(),
               'num_workers': flags.num_workers,
               'visual_type':flags.visual_type,
               'mode':flags.mode,
               'momentum':flags.momentum,
               'grad_save':flags.grad_save,
               'threshold':flags.threshold,
               'grad_off_epoch':flags.grad_off_epoch,
               'earlystop':flags.earlystop,
               'patience':flags.patience,
               }
    if configs['log_extraction'] == 'true' and configs['mode'] in train_mode_list:
        print("SEED:",flags.seed)
        save_params(configs, time_data)
        print("Using device: {}, Mode:{}, Type:{}".format(device,flags.mode,flags.nn_type))
        sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'grad_data','log_{}.txt'.format(time_data)),'w')
    else:
        if flags.file_name is not None:
            CALL_CONFIG = load_params(configs, file_name)
            CALL_CONFIG['visual_mode']=CALL_CONFIG['mode'] # training종류 선택
            CALL_CONFIG['mode']=configs['mode']
            if configs['mode']=='visual':
                CALL_CONFIG['visual_type']=configs['visual_type']
                print(configs['visual_type'])
            configs=CALL_CONFIG
            print("Mode:{}, Type:{}".format(configs['mode'],configs['nn_type']))
    
    #Visual
    if flags.mode == 'visual':
        from visualization import visualization
        configs = visualization(configs, file_name)
    else:
        from NeuralNet.baseNet import BaseNet
        model=BaseNet(configs).model
    
    #Train
    file_path=os.path.dirname(os.path.abspath(__file__))
    if flags.mode == 'train' or flags.mode=='train_weight_prune':
        from Learner.train import ClassicLearner
        learner=ClassicLearner(model,time_data,file_path,configs)
        configs=learner.run()
    elif flags.mode=='train_lrp' or flags.mode=='train_grad_visual':
        from Learner.gradprune import GradPruneLearner
        learner=GradPruneLearner(model,time_data,file_path,configs)
        configs=learner.run()
        save_params(configs, time_data)
    elif flags.mode=='train_mtl' or flags.mode=='train_mtl_v2':
        from Learner.mtl import MTLLearner
        learner=MTLLearner(model,time_data,file_path,configs)
        configs=learner.run()
        save_params(configs, time_data)

    
    print("End the process")
    if flags.log.lower()=='true':
        sys.stdout.close()

if __name__ == '__main__':
    main(sys.argv[1:])
