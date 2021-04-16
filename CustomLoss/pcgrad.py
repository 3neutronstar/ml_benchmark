import enum
import torch
import torch.nn as nn
import copy
import random
import numpy as np


class PCGrad(): # mtl_v2 only
    def __init__(self, optimizer):
        self._optim = optimizer
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

    def pc_backward(self, objectives,labels):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g[shared].to('cpu')
                                           for g in pc_grad]).mean(dim=0).cuda() #평균
        merged_grad[~shared] = torch.stack([g[~shared].to('cpu')
                                            for g in pc_grad]).sum(dim=0).cuda()
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

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

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

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class PCGrad_v2(PCGrad):
    def __init__(self,optimizer):
        super(PCGrad_v2,self).__init__(optimizer)
        self.objectives=list()

    @property
    def optimizer(self):
        return self._optim

    def step(self):

        grads, shapes, has_grads = self._pack_grad(self.objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad=self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        self._optim.step()
        self.objectives=list()
        return 

    def pc_backward(self, objectives,labels):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        
        self.objectives.append(objectives)
        return


class PCGrad_v3(PCGrad):
    def __init__(self,optimizer):
        super(PCGrad_v3,self).__init__(optimizer)
        self.objectives=None
        self.shape=[]
        for group in self._optim.param_groups:
            for p in group['params']:
                self.shape.append(p.shape)
        self.i=0
        self.batch_size=0

    @property
    def optimizer(self):
        return self._optim

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad=[]
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                grad.append(p.grad.clone())
        return grad
    
    def _project_conflicting(self, grads):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.stack([g for g in pc_grad]).mean(dim=0).cuda() #평균
        return merged_grad
        
    def step(self):
        objectives=copy.deepcopy(self.objectives)
        for i, i_obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            i_obj.backward(retain_graph=True)
            grad = self._retrieve_grad()
            g_i=self._flatten_grad(grad, self.shape)
            if i==0:
                pc_grad=torch.zeros_like(g_i)
            for j_obj in self.objectives:
                self._optim.zero_grad(set_to_none=True)
                j_obj.backward(retain_graph=True)
                grad = self._retrieve_grad()
                g_j=self._flatten_grad(grad, self.shape)

                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
            pc_grad=pc_grad+g_i

        pc_grad=self._unflatten_grad(pc_grad, self.shape)
        self._set_grad(pc_grad)
        self._optim.step()
        self.objectives=None
        self.i=0
        return 

    def pc_backward(self, objectives,labels):
        self.objectives=objectives
        self.batch_size=len(labels)
        return
    