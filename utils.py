import json
import os
import sys
import torch
import numpy as np
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

    def __init__(self, file_path,time_data,config,patience=7, verbose=False, delta=0,):
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
        self.config=config
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(file_path,'grad_data','checkpoint_{}.pt'.format(time_data))

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:

            self.counter += 1
            if self.config['earlystop']==True:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            print(
                f'Eval loss not decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(
                f'Eval loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class TestPerformance():
    def __init__(self,model,file_name,file_path,configs):
        self.model=model
        model.load_state_dict(torch.load(os.path.join(file_path,'grad_data','checkpoint_'+file_name+'.pt')))
        self.device=configs['device']
        model.to(self.device)

        self.file_path=file_path
        from DataSet.data_load import data_loader
        _,self.test_loader=data_loader(configs)
        self.criterion = self.model.loss
        self.configs=configs


    def run(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        class_correct_dict=dict()
        class_total_dict=dict()
        for i in range(self.configs['num_classes']):
            class_correct_dict[i]=0
            class_total_dict[i]=0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                eval_loss += loss.sum().item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                for label in target.unique():
                    # print(label,pred.eq(target.view_as(pred))[target==label].sum().item())
                    class_correct_dict[int(label)]+=pred.eq(target.view_as(pred))[target==label].sum().item()
                    class_total_dict[int(label)]+=(target==label).sum().item()

        eval_loss = eval_loss / len(self.test_loader.dataset)

        correct=0
        print("=================Eval=================")
        for class_correct_key in class_correct_dict.keys():
            correct+=class_correct_dict[class_correct_key]
            class_accur=100.0*float(class_correct_dict[class_correct_key])/class_total_dict[class_correct_key]
            print('{} class :{}/{} {:2f}%'.format(class_correct_key,class_correct_dict[class_correct_key],class_total_dict[class_correct_key],class_accur))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n==================='.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))
        eval_metric={'accuracy':eval_accuracy,'loss': eval_loss}

        return eval_metric
