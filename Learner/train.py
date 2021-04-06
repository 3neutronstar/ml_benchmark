import time
import os
import time
import numpy as np
import torch
from Learner.base_learner import BaseLearner
import sys

class ClassicLearner(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(ClassicLearner,self).__init__(model,time_data,file_path,configs)

        # pruning
        if configs['mode']=='train_weight_prune':
            self.grad_off_mask = list()
            self.grad_norm_cum = dict()
            for l, num_nodes in enumerate(self.model.node_size_list):
                for n in range(num_nodes):
                    self.grad_norm_cum['{}l_{}n'.format(l, n)] = 0.0
                self.grad_off_mask.append(torch.zeros(
                    num_nodes, dtype=torch.bool, device=self.device))  # grad=0으로 끄는 mask
            # gradient 꺼지는 빈도확인
            self.grad_off_freq_cum = 0
            # 꺼지는 시기
            self.grad_turn_off_epoch = self.configs['grad_off_epoch']
            # # 다시 켤 노드 지정
            self.grad_turn_on_dict=None
            # self.grad_turn_on_dict = {
            #     2: [0, 31, 58, 68, 73]
            #     # 3:[2,12,27,31,50,82]
            # }
            print(self.grad_turn_on_dict)


    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        # Train
        for epoch in range(1, self.configs['epochs'] + 1):
            train_accuracy, train_loss = self._train(epoch)
            eval_accuracy, eval_loss = self._eval()
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
            #TODO REMOVE for checking lateral inhibition
            if epoch ==1:
                break


        if self.configs['mode'] == 'train_weight_prune':
            print("before prune")
            for layer in self.optimizer.param_groups[0]['params']:
                print(layer.size())

            print("after prune")
            for mask_layer in self.grad_off_mask:
                print("Pruned weight", torch.nonzero(mask_layer).size())

            for layer in self.optimizer.param_groups[0]['params']:
                print("After Weight Prune", torch.nonzero(layer).size())

        configs = self.save_grad(epoch)
        return configs

    def _train(self, epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        correct = 0
        num_training_data = len(self.train_loader.dataset)
        # defalut is mean of mini-batchsamples, loss type설정
        # loss함수에 softmax 함수가 포함되어있음
        # 몇개씩(batch size) 로더에서 가져올지 정함 #enumerate로 batch_idx표현
        p_groups=self.optimizer.param_groups
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(
                self.device)  # gpu로 올림
            self.optimizer.zero_grad()  # optimizer zero로 초기화
            # weight prune #TODO
            self.prune_weight(p_groups,epoch,batch_idx)
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
            self._prune_grad(p_groups, epoch, batch_idx)
            # grad save(prune후 save)
            self._save_grad(p_groups, epoch, batch_idx)
            # prune 이후 optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.item()), end='')

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
            
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        return running_accuracy, running_loss

    def _eval(self):
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

#########################################################################################################

###########################################################################################################

    def _prune_grad(self, p_groups, epoch, batch_idx):
        # pruning mask generator
        l = -1  # 처음 layer는 0으로 증가해서 maxpooling과 같은 요소를 피하기 위함
        if self.configs['mode'] == 'train_weight_prune':
            for p in p_groups:
                for i, p_layers in enumerate(p['params']):
                    # first and last layer live
                    # or p['params'][0].size()==p_layers.size() or p['params'][1].size()==p_layers.size(): # 마지막 layer는 output이므로 배제
                    if p['params'][-1].size() == p_layers.size() or p['params'][-2].size() == p_layers.size():
                        continue
                    else:
                        # weight filtering
                        if len(p_layers.size()) > 1 and epoch <= self.grad_turn_off_epoch+1:
                            l += 1  # layer
                            p_nodes = p_layers.grad.cpu().detach().clone()  # prune in grad
                            # p_nodes = p_layers.grad.cpu().detach().clone()  # prune in weight
                            for n, p_node in enumerate(p_nodes):
                                # 1. gradient cumulative값이 일정 이하이면 모두 gradient prune
                                if epoch < self.grad_turn_off_epoch+1:
                                    self.grad_norm_cum['{}l_{}n'.format(
                                        l, n)] += p_node.norm().view(-1)  # cumulative value

                            p_layers.to(self.device)

            # pruning the gradient
            if epoch > self.grad_turn_off_epoch:
                l = -1  # -1부터해서 0으로 시작하게함
                for p in p_groups:
                    for i, p_layers in enumerate(p['params']):
                        if len(p_layers.size()) > 1:  # weight filtering
                            l += 1  # layer
                            # print(p_layers.data[self.grad_off_mask[l]].sum(),'a')
                            p_layers.grad[self.grad_off_mask[l]] = torch.zeros_like(
                                p_layers.grad[self.grad_off_mask[l]])  # weight prune
                        else:
                            p_layers.grad[self.grad_off_mask[l]] = torch.zeros_like(
                                p_layers.grad[self.grad_off_mask[l]])  # bias prune

    def turn_requires_grad_(self,p_groups,on_off):
        if self.configs['mode']=='train_weight_prune':
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    p_layers.requires_grad_(on_off)

    def prune_weight(self, p_groups,epoch,batch_idx):
        if self.configs['mode'] == 'train_weight_prune' and epoch >= self.grad_turn_off_epoch+1:
            l = -1  # -1부터해서 0으로 시작하게함, for bias로 여겨지는 avgpooling,maxpooling회피용
            #turn off judgement
            if epoch == self.grad_turn_off_epoch+1 and batch_idx == 0:
                for p in p_groups:
                    for i,p_layers in enumerate(p['params']):
                        if p['params'][-1].size() == p_layers.size() or p['params'][-2].size() == p_layers.size():
                            continue
                        if len(p_layers.size())>1: #weight filtering
                            l+=1
                            for n, p_node in enumerate(p_layers):
                                if self.grad_norm_cum['{}l_{}n'.format(l, n)] < self.configs['threshold']:
                                    self.grad_off_mask[l][n] = True
                                    print('{}l_{}n grad_off'.format(l, n))
                                    self.grad_off_freq_cum += 1

            # 끌필요없는 것 다시 켜는 것
            if epoch == self.grad_turn_off_epoch+1 and batch_idx == 0 and self.grad_turn_on_dict is not None:
                print("Turn on the designated node=====")
                for l_key in self.grad_turn_on_dict.keys():
                    for n in self.grad_turn_on_dict[l_key]:
                        self.grad_off_mask[l_key][n] = False
                        print("{}l_{}n grad Turn on and No Prune".format(l_key, n))
                print("End Turn on=====================")

            # Record Prune Rate
            self.logWriter.add_scalar(
                'train/grad_off_freq_cum', self.grad_off_freq_cum, epoch)
            self.turn_requires_grad_(p_groups,on_off=False)

            #weight prune
            l = -1  # -1부터해서 0으로 시작하게함, for bias로 여겨지는 avgpooling,maxpooling회피용
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    if len(p_layers.size())>1: #weight filtering
                        l+=1 #layer
                        # print(p_layers.data[self.grad_off_mask[l]].sum(),'b')
                        p_layers.data[self.grad_off_mask[l]]=torch.zeros_like(p_layers.data[self.grad_off_mask[l]])
                    else:# bias
                        p_layers.data[self.grad_off_mask[l]]=torch.zeros_like(p_layers.data[self.grad_off_mask[l]])
                        #print(l,"layer",torch.nonzero(p_layers.grad).size()," ",p_layers.grad.size())            
            self.turn_requires_grad_(p_groups,on_off=True)
    