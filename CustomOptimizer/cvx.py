import torch
import cvxpy as cp
import numpy as np

# class CVXOptimizer():
#     def __init__(self,optimizer):
#         self._optim=optimizer

    
#     @property
#     def optimizer(self):
#         return self._optim

#     def zero_grad(self):
#         '''
#         clear the gradient of the parameters
#         '''

#         return self._optim.zero_grad(set_to_none=True)

#     def step(self):
#         '''
#         update the parameters with the gradient
#         '''

#         return self._optim.step()


#     def cvx_backward(self,objectives):
#         cp_loss=objectives.data.cpu().numpy()
        
#         alpha_size=objectives.size()[0]
#         alpha=cp.Variable((alpha_size),nonneg=True)
#         alpha.value=torch.rand(alpha_size).numpy()
#         objective=cp.Minimize(cp.square(cp.abs(cp.sum(cp_loss@alpha))))
#         constraints=[0<=alpha,alpha<=1,sum(alpha)==1.0]
#         prob = cp.Problem(objective, constraints)
#         prob.solve()
#         # print(cp_loss@alpha.value,sum(alpha.value))
#         alpha_tensor=torch.tensor(alpha.value,dtype=torch.float,device='cuda').view(1,-1)
#         cvx_loss=torch.matmul(alpha_tensor,objectives)
#         cvx_loss.backward()      

#         return


class CVXOptimizer():
    def __init__(self,optimizer):
        self._optim=optimizer

    
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
    
    def _solve_cvx(self,grads):
        alpha=len(grads)
        alpha=cp.Variable((alpha),nonneg=True)
        alpha.value=torch.rand(len(grads)).numpy()
        cp_grads=torch.cat(grads,dim=0)
        objective=cp.Minimize(cp.square(cp.abs(cp.sum(cp_grads.T.cpu().numpy()@alpha))))
        constraints=[0<=alpha,alpha<=1,sum(alpha)==1.0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        alpha_tensor=torch.tensor(alpha.value,dtype=torch.float,device='cuda').view(1,-1)
        cvx_grad=torch.matmul(alpha_tensor,cp_grads).view(-1)
        
        return cvx_grad

    def cvx_backward(self,objectives):
        grads, shapes = self._pack_grad(objectives)
        cvx_grad=self._solve_cvx(grads)
        cvx_grad = self._unflatten_grad(cvx_grad, shapes[0])
        self._set_grad(cvx_grad)    
        return

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

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads]).view(1,-1)
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

