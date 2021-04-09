import torch
import torch.nn as nn
import copy
import random
import numpy as np

class PCGrad():
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

        grads, shapes, has_grads = self._pack_grad(objectives,labels)
        total_grads=list()
        for label_idx,grad in enumerate(grads):
            if len(has_grads[label_idx])==0:
                continue
            pc_grad = self._project_conflicting(grad, has_grads[label_idx])# 각 클래스당 surgery
            pc_grad = self._unflatten_grad(pc_grad, shapes[label_idx][0])# 각 클래스당 surgery
            total_grads.append(pc_grad)
        # 여기까지 각 label에 대한 grad 설정
        self._set_grad(total_grads)

        #summation
        #그냥진행
        #avg
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad=torch.div(p.grad,len(labels))
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
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, total_grads):
        '''
        set the modified gradients to the network
        '''
        #scalarization
        idx = 0
        for label_idx,total_grad in enumerate(total_grads):
            for group in self._optim.param_groups:
                for p,grad in zip(group['params'],total_grad):
                    # if p.grad is None: continue
                    if idx==0:
                        p.grad=torch.zeros_like(p.grad)
                p.grad += grad[idx].cuda()
                idx += 1
        return

    def _pack_grad(self, objectives,labels):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grads, shapes, has_grads = [[] for _ in set(labels.tolist())], [[] for _ in set(labels.tolist())], [[] for _ in set(labels.tolist())]
        for obj, label in zip(objectives,labels):
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads[label].append(self._flatten_grad(grad, shape))
            has_grads[label].append(self._flatten_grad(has_grad, shape))
            shapes[label].append(shape)

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
                p=p.to('cpu')
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

