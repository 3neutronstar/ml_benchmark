import time
import os
import time
import numpy as np
import torch
from base_learner import BaseLearner
import sys

class ClassicLearner(BaseLearner):
    def __init__(self, model, time_data, config):
        super(ClassicLearner,self).__init__(model,time_data,config)
        # grad list
        self.grad_list = list()

        # pruning
        self.grad_off_mask = list()
        self.mask = dict()
        self.grad_norm_dict = dict()
        self.grad_norm_cum = dict()
        for l, num_nodes in enumerate(self.model.node_size_list):
            for n in range(num_nodes):
                self.grad_norm_dict['{}l_{}n'.format(l, n)] = list()
                self.grad_norm_cum['{}l_{}n'.format(l, n)] = 0.0
                self.mask['{}l_{}n'.format(l, n)] = 0
            self.grad_off_mask.append(torch.zeros(
                num_nodes, dtype=torch.bool, device=self.device))  # grad=0으로 끄는 mask
            # True면 꺼짐

        # gradient 꺼지는 빈도확인
        self.grad_off_freq_cum = 0

        # 꺼지는 시기
        self.grad_turn_off_epoch = self.config['grad_off_epoch']

        # # 다시 켤 노드 지정
        self.grad_turn_on_dict=None
        # self.grad_turn_on_dict = {
        #     2: [0, 31, 58, 68, 73]
        #     # 3:[2,12,27,31,50,82]
        # }
        print(self.grad_turn_on_dict)

    def run(self):
        print("Training {} epochs".format(self.config['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        grad_list = list()
        # Train
        for epoch in range(1, self.config['epochs'] + 1):
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

        if self.config['mode'] == 'train_prune':
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
            if epoch==2 and batch_idx==0:
                print("hi")
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
        if self.config['log_extraction']=='true':
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
    def save_grad(self, epochs):
        # Save all grad to the file
        self.config['end_epoch'] = epochs
        if self.config['grad_save'] == 'true':
            param_size = list()
            params_write = list()

            tik = time.time()
            if self.config['nn_type'] == 'lenet300_100' or self.config['nn_type']=='lenet5':#lenet300 100
                for t, params in enumerate(self.grad_list):
                    if t == 1:
                        for i, p in enumerate(params):  # 각 layer의 params
                            param_size.append(p.size())
                    # elem
                    params_write.append(torch.cat(params, dim=0).unsqueeze(0))
                    # node

                    if t % 100 == 0:
                        print("\r step {} done".format(t), end='')
                        
            # elif self.config['nn_type'] == 'lenet5': #TODO
            #     for t, params in enumerate(self.grad_list):
            #         if t == 1:
            #             for i, p in enumerate(params):  # 각 layer의 params
            #                 param_size.append(p.size())
            #         # elem
            #         # print(params)
            #         params_write.append(torch.cat(params, dim=0).unsqueeze(0))
            #         # node

            #         if t % 100 == 0:
            #             print("\r step {} done".format(t), end='')

            else:  # vgg16
                import platform
                for epoch in range(1,epochs+1):
                    i = 0
                    epoch_data = list()
                    # check exist
                    while os.path.exists(os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))) == True:
                        batch_idx_data = np.load(os.path.join(
                            self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i)))
                        epoch_data.append(torch.from_numpy(batch_idx_data))
                        i += 1

                    params_write.append(torch.cat(epoch_data, dim=0))
                    print("\r {}epoch processing done".format(epoch),end='')
                print("\n")

            write_data = torch.cat(params_write, dim=0)
            if self.config['nn_type'] != 'lenet300_100' and self.config['nn_type']!='lenet5':
                for epoch in range(1,epochs+1):
                    i = 0
                    epoch_data = list()
                    # check exist
                    while os.path.exists(os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))) == True:
                        # remove
                        if platform.system() == 'Windows':
                            os.system('del {}'.format(os.path.join(
                                self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))))
                        else:
                            os.system('rm {}'.format(os.path.join(
                                self.making_path, 'tmp', '{}_{}e_{}.npy'.format(self.time_data, epoch, i))))
                        i+=1
                    print("\r {}epoch processing done".format(epoch),end='')
            print("\n Write data size:", write_data.size())
            np.save(os.path.join(self.making_path, 'grad_{}'.format(
                self.time_data)), write_data.numpy())  # npy save
            tok = time.time()
            print('play_time for saving:', tok-tik, "s")
            print('size: {}'.format(len(params_write)))

            '''
            Save params
            '''
        return self.config

    def _save_grad(self, p_groups, epoch, batch_idx):
        # save grad to the list
        if self.config['grad_save'] == 'true':
            save_grad_list = list()
            for p in p_groups:
                for l, p_layers in enumerate(p['params']):
                    
                    # node, rest
                    if self.config['nn_type'] == 'lenet300_100' or self.config['nn_type']=='lenet5':
                        if len(p_layers.size()) > 1:  # weight filtering
                            p_nodes = p_layers.grad.cpu().detach().clone()
                            # print(p_nodes.size())
                            for n, p_node in enumerate(p_nodes):
                                self.grad_list[-1].append(torch.cat([p_node.mean().view(-1), p_node.norm(
                                ).view(-1), torch.nan_to_num(p_node.var()).view(-1)], dim=0).unsqueeze(0))
                    # elif self.config['nn_type'] == 'lenet5':#TODO
                    #     if len(p_layers.size()) > 1:  # weight filtering
                    #         p_node = p_layers.grad.view(
                    #             -1).cpu().detach().clone()
                    #         # if i==0:
                    #         #     print(p_node[50:75])
                    #         #     print(p_node.size())
                    #         self.grad_list[-1].append(p_node)

                    else:  # vgg
                        if len(p_layers.size()) > 1:
                            p_nodes = p_layers.grad.cpu().detach().clone()
                            for n, p_node in enumerate(p_nodes):
                                save_grad_list.append(torch.cat([p_node.mean().view(-1), p_node.norm(
                                ).view(-1), torch.nan_to_num(p_node.var()).view(-1)], dim=0).unsqueeze(0))

                    p_layers.to(self.device)
            if 'lenet' not in self.config['nn_type']:
                npy_path = os.path.join(self.making_path, 'tmp', '{}_{}e_{}.npy'.format(
                    self.time_data, epoch, batch_idx))
                row_data = torch.cat(save_grad_list, dim=0).unsqueeze(0)
                np.save(npy_path, row_data.numpy())
                del save_grad_list
                del row_data

    def revert_grad_(self, p_groups):
        if self.configs['mode'] == 'train_prune' and self.grad_off_mask.sum() > 0 and len(self.grad_list) != 0:
            for p in p_groups:
                for i, p_layers in enumerate(p['params']):
                    for p_nodes in p_layers:
                        for p_node in p_nodes:
                            p_node.grad = self.grad_list[-1][0]
                            self.grad_list[-1].pop(0)
                    self.grad_list.pop(-1)  # node뽑기

###########################################################################################################

    def _prune_grad(self, p_groups, epoch, batch_idx):
        # pruning mask generator
        l = -1  # 처음 layer는 0으로 증가해서 maxpooling과 같은 요소를 피하기 위함
        if self.config['mode'] == 'train_prune':
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
                            # if batch_idx == 0:
                            #     print(torch.nonzero(p_layers).size())
                            #     print(torch.nonzero(p_layers.grad).size())
                            #p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                        else:
                            p_layers.grad[self.grad_off_mask[l]] = torch.zeros_like(
                                p_layers.grad[self.grad_off_mask[l]])  # bias prune
                            # p_layers[self.grad_off_mask[l]]=torch.zeros_like(p_layers[self.grad_off_mask[l]])
                            # print(l,"layer",torch.nonzero(p_layers.grad).size()," ",p_layers.grad.size())

    def turn_requires_grad_(self,p_groups,on_off):
        if self.config['mode']=='train_prune':
            for p in p_groups:
                for i,p_layers in enumerate(p['params']):
                    p_layers.requires_grad_(on_off)

    def prune_weight(self, p_groups,epoch,batch_idx):
        if self.config['mode'] == 'train_prune' and epoch >= self.grad_turn_off_epoch+1:
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
                                if self.grad_norm_cum['{}l_{}n'.format(l, n)] < self.config['threshold']:
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
    