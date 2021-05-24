import torch
import copy
import random
import numpy as np

class LayerByLayerOptimizer():
    def __init__(self, model,optimizer):
        self._optim = optimizer
        self._model=model
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

    def backward(self,objectives,labels):

        pc_objectives=list()
        for idx in labels.unique():
            pc_objectives.append(objectives[labels==idx].mean().view(1))
        # pc_objectives=torch.cat(pc_objectives,dim=0)
        # pc_objectives.backward(retain_graph=True)
        #for i,(fhook,bhook) in enumerate(reversed(zip(self.hookForward,self.hookBackward))):
        #    if i==0:
        #        for pc_obj in pc_objectives:
        #            self._optim.zero_grad()
        #            pc_obj.backward(retain_graph=True)
        grads, shapes = self._pack_grad(pc_objectives)
        pc_grad = self._project_conflicting(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
               

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
        return grads, shapes


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


    def _flatten_grad(self, grads, shapes):
        flatten_grad = [g.flatten() for g in grads]
        return flatten_grad


    def _project_conflicting(self, grads, shapes=None,epoch=None,batch_idx=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)

        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                for g_l_i,g_l_j in zip(g_i,g_j):
                    g_i_g_j = torch.dot(g_l_i, g_l_j)
                    g_l_j_norm=(g_l_j.norm()**2)
                    if g_l_j_norm>1e-10 and g_i_g_j<-(1e-10): #g_i_g_j<0:
                            g_l_i -= (g_i_g_j) * g_l_j / g_l_j_norm
        merged_grad=[]
        for layer_idx,g_l_i in enumerate(pc_grad[0]):
            merged_grad.append(torch.cat([grad[layer_idx] for grad in pc_grad],dim=0).view(num_task,-1).mean(dim=0))
        return merged_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for grad,shape in zip(grads,shapes):
            length = np.prod(shape)
            unflatten_grad.append(grad.view(shape).clone())
            idx += length
        return unflatten_grad

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