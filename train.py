import time
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from six.moves import urllib
from utils import EarlyStopping
from DataSet.data_load import data_loader

class Learner():
    def __init__(self,model,time_data,config):
        self.model=model
        self.optimizer=self.model.optim
        self.criterion=self.model.loss
        self.scheduler=self.model.scheduler
        self.config=config
        self.grad_list=list()
        self.log_interval=100
        self.device = self.config['device']
        # data
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        self.train_loader, self.test_loader = data_loader(self.config)
        # Tensorboard
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.logWriter = SummaryWriter(os.path.join(
            self.current_path, 'training_data', time_data))
        self.time_data=time_data

        self.early_stopping = EarlyStopping(self.current_path,time_data,patience = self.config['patience'], verbose = True)

        # grad list
        self.grad_list=list()

        #pruning
        self.grad_off_mask=list()
        self.mask=dict()
        self.grad_norm_dict=dict()
        self.grad_norm_cum=dict()
        for l,num_nodes in enumerate(self.model.b_size_list):
            for n in range(num_nodes):
                self.grad_norm_dict['{}l_{}n'.format(l,n)]=list()
                self.grad_norm_cum['{}l_{}n'.format(l,n)]=0.0
                self.mask['{}l_{}n'.format(l,n)]=0
            self.grad_off_mask.append(torch.zeros(num_nodes,dtype=torch.bool,device=self.device))# grad=0으로 끄는 mask
            # True면 꺼짐
        
        # gradient 꺼지는 빈도확인
        self.grad_off_freq_cum=0


    def run(self):
        print("Training {} epochs".format(self.config['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        grad_list=list()
        # Train
        for epoch in range(1, self.config['epochs'] + 1):
            train_accuracy, train_loss = self.train_(epoch)
            eval_accuracy, eval_loss = self.eval_()
            self.scheduler.step()
            loss_dict = {'train': train_loss, 'eval': eval_loss}
            accuracy_dict = {'train': train_accuracy, 'eval': eval_accuracy}
            self.logWriter.add_scalars('loss', loss_dict, epoch)
            self.logWriter.add_scalars('accuracy', accuracy_dict, epoch)

            self.early_stopping(eval_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if self.device == 'gpu':
                torch.cuda.empty_cache()

        if self.config['mode']=='train_prune':
            print("before prune")
            for layer in self.optimizer.param_groups[0]['params']:
                print(layer.size())

            print("after prune")
            for mask_layer in self.grad_off_mask:
                print("Pruned rate",torch.nonzero(mask_layer).size())

            for layer in self.optimizer.param_groups[0]['params']:
                print("Weight Prune", torch.nonzero(layer).size())
            
        configs=self.save_grad()
        return configs
    
    def train_(self,epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        correct = 0
        num_training_data = len(self.train_loader.dataset)
        # defalut is mean of mini-batchsamples, loss type설정
        # loss함수에 softmax 함수가 포함되어있음
        # 몇개씩(batch size) 로더에서 가져올지 정함 #enumerate로 batch_idx표현
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)  # gpu로 올림
            self.optimizer.zero_grad()  # optimizer zero로 초기화
            # model에서 입력과 출력이 나옴 batch 수만큼 들어가서 batch수만큼 결과가 나옴 (1개 인풋 1개 아웃풋 아님)
            output = self.model(data)
            loss = self.criterion(output, target)  # 결과와 target을 비교하여 계산
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()  # 역전파
            p_groups = self.optimizer.param_groups  # group에 각 layer별 파라미터
            self.grad_list.append([])
            # grad prune
            self.prune_grad_(p_groups,epoch,batch_idx)     
            # grad save
            self.save_grad_(p_groups)
            # prune 이후 optimizer step
            self.optimizer.step()
            # weight prune
            self.prune_weight(p_groups,epoch)
                            
            running_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.item()), end='')

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        return running_accuracy, running_loss

    def eval_(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        criterion = self.model.loss  # add all samples in a mini-batch
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                eval_loss += loss.item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss = eval_loss / len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))

        return eval_accuracy, eval_loss
    
    def save_grad(self):
        # Save all grad to the file 
        if self.config['log_extraction']==True:
            param_size = list()
            params_write = list()
            if self.config['colab'] == True:
                making_path = os.path.join('drive', 'MyDrive', 'grad_data')
            else:
                making_path = os.path.join(self.current_path, 'grad_data')
            if os.path.exists(making_path) == False:
                os.mkdir(making_path)
            tik = time.time()
            import numpy as np
            if self.config['log_extraction'] == True:
                if self.config['nn_type']=='lenet5':
                    for t, params in enumerate(self.grad_list):
                        if t == 1:
                            for i, p in enumerate(params):  # 각 layer의 params
                                param_size.append(p.size())
                        #elem
                        # print(params)
                        params_write.append(torch.cat(params,dim=0).unsqueeze(0))
                        #node

                        if t % 100 == 0:
                            print("\r step {} done".format(t), end='')
                    write_data = torch.cat(params_write, dim=0)
                else: # lenet300 100 # vgg16
                    for t, params in enumerate(self.grad_list):
                        if t == 1:
                            for i, p in enumerate(params):  # 각 layer의 params
                                param_size.append(p.size())
                        #elem
                        params_write.append(torch.cat(params,dim=0).unsqueeze(0))
                        #node

                        if t % 100 == 0:
                            print("\r step {} done".format(t), end='')
                    write_data = torch.cat(params_write, dim=0)

                print("\n Write data size:", write_data.size())
                np.save(os.path.join(making_path, 'grad_{}'.format(
                    self.time_data)), write_data.numpy())#npy save
                tok = time.time()
                print('play_time for saving:', tok-tik, "s")
                print('size: {}'.format(len(params_write)))

            '''
            Save params
            '''
        return self.config


    def save_grad_(self,p_groups):
        # save grad to the list
        if self.config['log_extraction']==True:
            for p in p_groups:
                for l,p_layers in enumerate(p['params']):
                    if self.config['nn_type']=='lenet5':# or config['nn_type']=='lenet300_100':
                        if len(p_layers.size())>1: #weight filtering
                            p_node=p_layers.grad.view(-1).cpu().detach().clone()
                            # if i==0:
                            #     print(p_node[50:75])
                            #     print(p_node.size())
                            self.grad_list[-1].append(p_node)
                    # node, rest
                    else:
                        if len(p_layers.size())>1: #weight filtering
                            p_nodes=p_layers.grad.cpu().detach().clone()
                            # print(p_nodes.size())
                            for n,p_node in enumerate(p_nodes):
                                self.grad_list[-1].append(torch.cat([p_node.mean().view(-1),p_node.norm(2).view(-1),torch.nan_to_num(p_node.var()).view(-1)],dim=0).unsqueeze(0))
                    p_layers.to(self.device)


    def prune_grad_(self,p_groups,epoch,batch_idx):
        # pruning mask generator
        grad_turn_off_epoch=5
        if self.config['mode']=='train_prune':
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    if len(p_layers.size())>1 and epoch<=grad_turn_off_epoch+1: #weight filtering
                        l=int(i/2)
                        p_nodes=p_layers.grad.cpu().detach().clone()
                        for n,p_node in enumerate(p_nodes):
                            #1. gradient cumulative값이 일정 이하이면 모두 gradient prune
                            if epoch<grad_turn_off_epoch+1:
                                self.grad_norm_cum['{}l_{}n'.format(l,n)]+=p_node.norm(2).view(-1) # cumulative value
                            if epoch ==grad_turn_off_epoch+1 and batch_idx==0:
                                if self.grad_norm_cum['{}l_{}n'.format(l,n)]<200: # 100 이하면
                                    self.grad_off_mask[l][n]=True
                                    print('{}l_{}n grad_off'.format(l,n))
                                    self.grad_off_freq_cum+=1

                            # #2. gradient의 gradient threshold 이후 종료
                            # if epoch >5 and self.grad_off_mask[l][n]==False:
                            #     self.grad_norm_dict['{}l_{}n'.format(l,n)].append(p_node.norm(2).view(-1))
                            #     if len(self.grad_norm_dict['{}l_{}n'.format(l,n)])>1:
                            #         # # 2-1. gradient 값의 norm 기준 prune
                            #         # if self.grad_norm_dict['{}l_{}n'.format(l,n)][-1]<1e-4:
                            #         #     self.mask['{}l_{}n'.format(l,n)]+=1
                            #         # # 2-2. difference of gradient norm 기준 prune
                            #         # if self.grad_norm_dict['{}l_{}n'.format(l,n)][-1]-self.grad_norm_dict['{}l_{}n'.format(l,n)][-2]<1e-7:
                            #         #     self.mask['{}l_{}n'.format(l,n)]+=1
                            #         # self.grad_norm_dict['{}l_{}n'.format(l,n)].pop(0)

                            #     if self.mask['{}l_{}n'.format(l,n)]>100:
                            #         self.grad_off_mask[l][n]=True
                            #         self.grad_off_freq_cum+=1
                            #         print('{}epoch {}iter {}l_{}n grad_off'.format(epoch,batch_idx,l,n))

                            # #3. gradient의 moving average threshold 이후 종료
                            # if epoch >5 and self.grad_off_mask[l][n]==False:
                            #     self.grad_norm_cum['{}l_{}n'.format(l,n)]+=p_node.norm(2).view(-1) # moving avg 하셈
                            #     if self.grad_norm_cum['{}l_{}n'.format(l,n)]<1e-4: # moving avg가 일정 이하
                            #         self.grad_off_mask[l][n]=True
                            #         print('{}epoch {}iter {}l_{}n grad_off'.format(epoch,batch_idx,l,n))
                            #         self.grad_off_freq_cum+=1

                        p_layers.to(self.device)
            
            # Record Prune Rate
            self.logWriter.add_scalar('train/grad_off_freq_cum',self.grad_off_freq_cum,epoch)
            # pruning the gradient
            if epoch >grad_turn_off_epoch:
                for p in p_groups:
                    for i,p_layers in enumerate(p['params']):
                        if len(p_layers.size())>1: #weight filtering
                            l=int(i/2)
                            p_layers.grad[self.grad_off_mask[l]]=0.0#weight prune
                            # p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                        else:
                            p_layers.grad[self.grad_off_mask[l]]=0.0 #bias prune
                            # p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                            #print(l,"layer",torch.nonzero(p_layers.grad).size()," ",p_layers.grad.size())
    
    def turn_requires_grad_(self,p_groups,on_off):
        if self.config['mode']=='train_prune':
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    p_layers.requires_grad_(on_off)
    
    def save_prune_weight(self,p_groups,epoch):
        if self.config['mode']=='train_prune' : 
            if epoch > 5:
                for p in p_groups:
                    for i,p_layers in enumerate(p['params']):
                        p_layers.requires_grad_(False)
                        if len(p_layers.size())>1: #weight filtering
                            l=int(i/2)
                            p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                        else:# bias
                            p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                
        

    def prune_weight(self,p_groups,epoch):
        if self.config['mode']=='train_prune' :
            if epoch >5:
                self.turn_requires_grad_(p_groups,on_off=False)
                for p in p_groups:
                    for i,p_layers in enumerate(p['params']):
                        p_layers.requires_grad_(False)
                        if len(p_layers.size())>1: #weight filtering
                            l=int(i/2)
                            p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                        else:# bias
                            p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                            #print(l,"layer",torch.nonzero(p_layers.grad).size()," ",p_layers.grad.size())            
                self.turn_requires_grad_(p_groups,on_off=True)


    def revert_grad_(self,p_groups):
        if self.configs['mode']=='train_prune' and self.grad_off_mask.sum()>0 and len(self.grad_list)!=0:
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    for p_nodes in p_layers:
                        for p_node in p_nodes:
                            p_node.grad=self.grad_list[-1][0]
                            self.grad_list[-1].pop(0)
                    self.grad_list.pop(-1)#node뽑기
                