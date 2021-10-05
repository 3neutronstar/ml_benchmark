import torch
import copy
import random
import numpy as np
import math
class LayerByLayerOptimizer():
    def __init__(self, model,optimizer):
        self._optim = optimizer
        self._model=model
        self.phi_hat_ijk=None
        self.beta=1e-2
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

    def backward(self,objectives,labels,epoch=None):

        pc_objectives=list()
        if epoch>=0:
            self._optim.zero_grad()
            # before_grad=self._pack_grad(objectives.mean().view(-1))[0][0]
            # before_scale=before_grad.norm()
            # print("Before",self._pack_grad(objectives.mean().view(-1))[0][0].norm())
            # for idx in labels.unique():
            #     pc_objectives.append(objectives[labels==idx].mean().view(1))
            # pc_objectives=torch.cat(pc_objectives,dim=0)
            # pc_objectives.backward(retain_graph=True)
            #for i,(fhook,bhook) in enumerate(reversed(zip(self.hookForward,self.hookBackward))):
            #    if i==0:
            #        for pc_obj in pc_objectives:
            #            self._optim.zero_grad()
            #            pc_obj.backward(retain_graph=True)
            #grads, shapes = self._pack_grad(pc_objectives)
            grads, shapes = self._pack_grad(objectives)
            # pc_grad = self._project_conflicting(grads)
            pc_grad = self._gradvac(grads)
            # print("after",pc_grad.norm())
            # print("cosine_similarity",torch.dot(before_grad,pc_grad)/(pc_grad.norm()*before_grad.norm()))
            pc_grad = self._unflatten_grad(pc_grad, shapes[0])
            self._set_grad(pc_grad)
        else:
            objectives.mean().backward()
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
                if torch.equal(group['params'][-1],p) or torch.equal(group['params'][-2],p):
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                # # original lbl
                # shape.append(p.grad.shape)
                # grad.append(p.grad.clone())
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

    def _gradvac(self, grads, shapes=None):
        num_task=len(grads)
        gradvac = copy.deepcopy(grads)
        idx =np.arange(num_task)
        if self.phi_hat_ijk==None:
            self.phi_hat_ijk=torch.zeros((num_task,num_task,len(grads[0])),dtype=torch.float,device=grads[0][0].device)
        # print(torch.cat([torch.cat(g,dim=0) for g in gradvac],dim=0).view(num_task,-1).mean(dim=0).mean(),'b')
        for i, g_i in enumerate(gradvac):
            random.shuffle(idx)
            for j in idx:
                g_j = grads[j]
                for k,(g_ik,g_jk) in enumerate(zip(g_i,g_j)):
                    phi_ijk = torch.dot(g_ik, g_jk) / (g_ik.norm() * g_jk.norm())
                    if phi_ijk < self.phi_hat_ijk[i,j,k]:
                        phi_hat_ijk=self.phi_hat_ijk[i,j,k]
                        gradvac[i][k] = g_ik + g_jk*(g_ik.norm()*(phi_hat_ijk*torch.sqrt(1-phi_ijk**2)-phi_ijk*torch.sqrt(1-phi_hat_ijk**2)) /(g_jk.norm()*torch.sqrt(1-phi_hat_ijk**2)))
                    self.phi_hat_ijk[i,j,k] = (1-self.beta)*self.phi_hat_ijk[i,j,k] + self.beta*phi_ijk
                # self.phi_hat_ijk[j,i,k] = self.phi_hat_ijk[i,j,k]
        merged_grad=[]
        for layer_idx, _ in enumerate(gradvac[0]):
            merged_grad.append(torch.cat([g[layer_idx] for g in gradvac],dim=0).view(num_task,-1).mean(dim=0))
        # print(torch.cat(merged_grad,dim=0).mean(),'a')
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
                if torch.equal(group['params'][-1],p) or torch.equal(group['params'][-2],p):
                    p.grad = grads[idx]
                    idx += 1
        return


class LayerByLayerOptimizer_V2(LayerByLayerOptimizer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _retrieve_grad(self):
        grad, shape = [], []
        for group in self._optim.param_groups:
            for i,p in enumerate(group['params']):
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                # if torch.equal(group['params'][-1],p) or torch.equal(group['params'][-2],p):# classifier만
                if torch.equal(group['params'][3],p) or  torch.equal(group['params'][4],p):# classifier만
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
        return grad,shape
    
    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for i,p in enumerate(group['params']):
                # if p.grad is None: continue
                # if torch.equal(group['params'][-1],p):
                #     p.grad = grads[1]
                if torch.equal(group['params'][3],p):
                    p.grad = grads[idx].clone()
                    idx += 1
                    # print(grads[0].norm(),'after',p.size())
                if torch.equal(group['params'][4],p):
                    p.grad = grads[idx].clone()
                    idx += 1
        return

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def _project_conflicting(self, grads, shapes=None,labels=None,epoch=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # original
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j<-1e-15:
                    # g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
                    g_i -= (g_i_g_j) * g_j / torch.matmul(g_j,g_j)

        merged_grad = torch.cat(pc_grad,dim=0).view(num_task,-1).mean(dim=0)
        return merged_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
