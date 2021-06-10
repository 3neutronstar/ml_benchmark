import enum
import torch
import torch.nn as nn
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class PCGrad(): # mtl_v2 only# cpu 안내리기
    def __init__(self, optimizer):
        self._optim = optimizer
        self.conflict_list=None
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives,labels,epoch=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, labels=labels, epoch=epoch)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return
    
    def _check_priority(self,my_grads,grads):
        similarity_grads=list()
        for grad in grads:
            similarity_grads.append((torch.dot(my_grads,grad)/grad.norm()).view(1,-1))
        
        sorted_idx=torch.argsort(torch.cat(similarity_grads,dim=1),descending=True).view(-1)
        #print(sorted_idx,similarity_grads)
        return sorted_idx

    def _check_cosine_similarity(self,grads,labels=None):
        if self.conflict_list is None:
            self.conflict_list=[[[]for i in range(10)]for i in range(10)]
        grads=list(enumerate(grads))
        for label_idx,g_i in enumerate(grads):
            for j,g_j in grads:
                if label_idx<=j:#자기 자신 및 중복 데이터 제외
                    continue
                cosine_similarity=torch.dot(g_i, g_j) / (g_i.norm()*g_j.norm())
                self.conflict_list[labels[label_idx]][labels[j]].append(cosine_similarity)

    def _project_conflicting(self, grads, shapes=None,labels=None,epoch=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)

        # # 1.
        # for g_i in pc_grad:
        #     sorted_idx=self._check_priority(g_i,grads)
        #     for j in sorted_idx:
        #         g_i_g_j = torch.dot(g_i, grads[j])
        #         if g_i_g_j < 0:
        #             if grads[j].norm()>1e-20:
        #                 g_i -= (g_i_g_j) * grads[j] / torch.matmul(grads[j],grads[j])
        #             # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)

        # 3. # original
        for g_i in pc_grad:
           random.shuffle(grads)
           for g_j in grads:
               g_i_g_j = torch.dot(g_i, g_j)
               if  g_i_g_j>-(1e-20):
                   # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
                   g_i += (g_i_g_j) * g_j / torch.matmul(g_j,g_j)

        merged_grad = torch.cat(pc_grad,dim=0).view(num_task,-1).mean(dim=0)
        
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes = [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape= self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        self._optim.zero_grad(set_to_none=True)
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
        return grad, shape
    


class PCGrad_v2(PCGrad):
    '''
        Non-serialized pc grad
    '''
    def __init__(self,optimizer):
        super(PCGrad_v2,self).__init__(optimizer)

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])#.view(1,-1)
        return flatten_grad

    def _project_conflicting(self, grads, shapes=None, labels=None, epoch=None):
        
        num_task = len(grads)
        pc_grad=torch.cat(grads,dim=0).view(num_task,-1)
        # random.shuffle(grads)
        # g_i,g_j=torch.cat(pc_grad,dim=0),torch.cat(grads,dim=0)        

        ## Vectorized version
        # index=[i for i in range(num_task)]# shuffle해서 사용
        # shuffle_index=list()
        # for i in range(num_task):
        #    random.shuffle(index)
        #    shuffle_index.append(copy.deepcopy(index))

        # # g_i[index]=g_j
        # shuffle_index=torch.tensor(shuffle_index)

        # for idx in range(num_task):
        #    index=shuffle_index[:,idx]
        #    this_g_j=g_j[index]
        #    g_j_g_i=torch.matmul(g_i,this_g_j.T)
        #    this_g_j_g_i=torch.diagonal(g_j_g_i)
        #    index_surgery=torch.bitwise_and(this_g_j_g_i<0,(this_g_j.norm(dim=1)>1e-10))
        
        #    g_i[index_surgery]-=torch.div(torch.mul(this_g_j_g_i.view(-1,1),this_g_j).T,(this_g_j.norm(dim=1)**2)).T[index_surgery]
        # merged_grad = g_i.mean(dim=0).view(-1)
        self._check_cosine_similarity(grads,labels)
        #check the confliction
        for label_idx,g_i in enumerate(pc_grad):
            random.shuffle(grads)
            for j,g_j in enumerate(grads):
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j<-(1e-20):
                    # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
                    g_i -= (g_i_g_j) / torch.matmul(g_j,g_j) * g_j


        merged_grad=pc_grad.mean(dim=0)
        return merged_grad


class PCGrad_MOO(PCGrad_v2):
    '''
    PC_GRAD for moo
    '''
    def __init__(self,optimizer):
        super(PCGrad_MOO,self).__init__(optimizer)

    def pc_backward(self, objectives, labels, epoch):
        pc_objectives=list()
        for idx in torch.unique(labels):
            pc_objectives.append(objectives[labels==idx].mean().view(1))
        super().pc_backward(pc_objectives, labels, epoch=epoch)

        return torch.cat(pc_objectives,dim=0)

class PCGrad_MOO_V2(PCGrad_v2):
    '''
    PC_GRAD for moo
    '''
    def __init__(self,optimizer):
        super(PCGrad_MOO_V2,self).__init__(optimizer)

    def pc_backward(self, objectives, labels, epoch):
        # pc_objectives=list()
        # for idx in torch.unique(labels):
        #     pc_objectives.append(objectives[labels==idx].mean().view(1))
        # super().pc_backward(pc_objectives, labels, epoch=epoch)
        super().pc_backward(objectives, labels, epoch=epoch)

        # return torch.cat(pc_objectives,dim=0)
        return objectives

    # def _project_conflicting(self, grads,shapes=None,labels=None, epoch=None):
    #     pc_grad, num_task = copy.deepcopy(grads), len(grads)
    #     # random.shuffle(grads)
    #     g_i,g_j=torch.cat(pc_grad,dim=0),torch.cat(grads,dim=0)        

    #     index=[i for i in range(num_task)]# shuffle해서 사용
    #     shuffle_index=list()
    #     for i in range(num_task):
    #        random.shuffle(index)
    #        shuffle_index.append(copy.deepcopy(index))

    #     # g_i[index]=g_j
    #     shuffle_index=torch.tensor(shuffle_index)

    #     for idx in range(num_task):
    #        index=shuffle_index[:,idx]
    #        this_g_j=g_j[index]
    #        g_j_g_i=torch.matmul(g_i,this_g_j.T)
    #        this_g_j_g_i=torch.diagonal(g_j_g_i)
    #        index_surgery=torch.bitwise_and(this_g_j_g_i<0,(this_g_j.norm(dim=1)>1e-10))
        
    #        g_i[index_surgery]-=torch.div(torch.mul(this_g_j_g_i.view(-1,1),this_g_j).T,(this_g_j.norm(dim=1)**2)).T[index_surgery]
        
    #     weight_grad= torch.bincount(labels).float()/float(torch.numel(labels))
    #     assert(weight_grad.size()!=labels.unique())
    #     merged_grad = torch.matmul(weight_grad[labels.unique()],g_i)
    #     return merged_grad

    def _project_conflicting(self, grads, shapes=None, labels=None, epoch=None):
        num_task = len(grads)
        # print(grads)
        pc_grad=torch.cat(grads,dim=0).view(num_task,-1)
        merged_grad=list()
        for label in labels.unique():
            label_idx=(label==labels).nonzero().view(-1).clone()
            for l_idx in label_idx:
                g_i=pc_grad[l_idx]
                shuffle_idx=torch.randperm(label_idx.size()[0])#변경점 (log false인거 기준)
                shuffle_label_idx=label_idx[shuffle_idx].clone()# 변경점
                for s_idx in shuffle_label_idx:
                    g_j=grads[s_idx]
                    g_i_g_j = torch.dot(g_i, g_j)
                    # if g_i_g_j<-(1e-20):
                    #     # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
                    #     g_i -= (g_i_g_j) / torch.matmul(g_j,g_j) * g_j
                    if g_i_g_j>(1e-20):
                        g_i += (g_i_g_j) / torch.matmul(g_j,g_j) * g_j
                        # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
            merged_grad.append(pc_grad[label_idx].mean(dim=0).view(1,-1))
        merged_pc_grad=torch.cat(merged_grad,dim=0).mean(dim=0)
        return merged_pc_grad

class PCGrad_MOO_Baseline(PCGrad):
    def __init__(self,optimizer):
        super(PCGrad_MOO_Baseline,self).__init__(optimizer)

    def pc_backward(self, objectives, labels, epoch):
        self._optim.zero_grad()
        objectives.backward()
        return

class PCGrad_MOO_Baseline_V2(PCGrad_v2):

    def __init__(self,optimizer):
        super().__init__(optimizer)

    def pc_backward(self, objectives, labels, epoch):
        self._optim.zero_grad()
        super().pc_backward(objectives, labels, epoch=epoch)
        return

    def _project_conflicting(self, grads, shapes, labels, epoch):
        num_task = len(grads)
        self._check_cosine_similarity(grads,labels)
        pc_grad=torch.cat(grads,dim=0).view(num_task,-1)
        del grads
        pc_grad=pc_grad.mean(dim=0)
        return pc_grad

