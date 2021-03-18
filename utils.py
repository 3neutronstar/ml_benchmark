import json
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

def load_params(configs, file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(current_path, 'grad_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def save_params(configs, time_data):
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, 'grad_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)

def load_model(model,file_path,file_name):
    model.load_state_dict(torch.load(os.path.join(file_path,'grad_data','checkpoint_'+file_name+'.pt')))
    return model

class EarlyStopping:
    """주어진 patience 이후로 train loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, file_path,time_data,patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): train loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 train loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(file_path,'grad_data','checkpoint_{}.pt'.format(time_data))

    def __call__(self, train_loss, model):

        score = train_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(
                f'Train loss not decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss,model)
            self.counter = 0
            
    def save_checkpoint(self, train_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.train_loss_min = train_loss

def hist(model, device, test_loader, logWriter,prune_or_not):
    if prune_or_not==True:
        ptf='prune'
    else:
        ptf='no_prune'
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for i,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output,feature = model.extract_feature(data)
            if i%1000==0:
                logWriter.add_histogram('feature_map/{}th_photo'.format(i),feature,i)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset),
        100.0 * correct / float(len(test_loader.dataset))))
    eval_accuracy = 100.0*correct/float(len(test_loader.dataset))

    return eval_accuracy, eval_loss


def channel_hist(model,file_path,file_name,configs):
    from DataSet.data_load import data_loader
    _,test_data_loader=data_loader(configs)
        # Tensorboard
    logWriter = SummaryWriter(os.path.join(
        file_path,'grad_data', file_name,'hist'))

    model=load_model(model,file_path,file_name)

    # no_prune_eval_accuracy, _ = hist(model, configs['device'], test_data_loader, logWriter,prune_or_not=False)
    prune_ln_dict={4:[4,6,7,8,11,16,22,25,26,31,33,37,38,40,41,47,48,53,54,55,57,58,63,64,66,73,74,76,83,84,89,96,100,106,109,110,116]}
    # pruning
    params=model.parameters()
    # print(model.conv3.weight[0].size())
    for l,p_layer in enumerate(params):
        if l==4:
            print("Before",torch.nonzero(p_layer).size())

    for ln_idx in prune_ln_dict[4]:
        model.conv3.weight[ln_idx]=0.0
    # for l,p_layer in enumerate(params):
    #     if l%2==0:
    #         for n,p_node in enumerate(p_layer):
    #             if l==4: # layer 2에서 CNN
    #                 # print(p_node,prune_ln_dict[l])
    #                 if n in prune_ln_dict[l]:
    #                     print(p_node.sum())
    #                     p_node=torch.zeros_like(p_node)
    #                     print(p_node.sum())
    
    params_a=model.parameters()
    for l,p_layer in enumerate(params_a):
        if l==4:
            print("After",torch.nonzero(p_layer).size())
    prune_eval_accuracy,_=hist(model,configs['device'],test_data_loader, logWriter,prune_or_not=False)
    print("Before Accur: {}% After Prune Accur: {}%".format(no_prune_eval_accuracy,prune_eval_accuracy))



