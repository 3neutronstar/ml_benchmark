import torch
import torch.optim as optim
class LateralInhibition():
    def __init__(self,optimizer:optim):
        self._optim=optimizer

    @property
    def optimizer(self):
        return self._optim

    def step(self):
        return self._optim.step()
    
    def backward(self,loss):
        grads, shapes, has_grads = self._pack_grad(loss)
        #grad = self._unflatten_grad(grads, shapes)
        #self._set_grad(grad)
        for group in self._optim.param_groups:
            for g_layers,p in zip(grads,group['params']):
                if len(g_layers.size()) > 1:  # weight filtering
                    if len(g_layers.size())==4: #CNN
                        b_matrix=self._lateral_inhibition(g_layers)

                    elif len(g_layers.size())==2: #FC
                        b_matrix=self._lateral_inhibition(g_layers)
                else: #Bias
                    b_matrix=self._lateral_inhibition(g_layers)
                p.grad=b_matrix.clone()

    def _lateral_inhibition(self,grad_layers):
        k,alpha,beta,num=2,1e-4,0.75,5
        b_matrix=torch.zeros_like(grad_layers)
        for n,g_node in enumerate(grad_layers):
            lower_idx=int(max(0,n-num/2))
            upper_idx=int(min(grad_layers.size()[0]-1,n+num/2))
            gain=torch.pow(k+alpha*torch.square(grad_layers[lower_idx:upper_idx]).sum(),beta)
            b_matrix[n]=torch.div(g_node,gain)
        return b_matrix 
      
    ### Original
    def _pack_grad(self, loss):
        self._optim.zero_grad(set_to_none=True)
        loss.backward()
        grad, shape, has_grad = self._retrieve_grad()
        return grad, shape, has_grad
     
    ### Original
    def _retrieve_grad(self):
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


