import torch
class LookUpGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        print('Instantiate Grad Profiler')

    @property
    def optimizer(self):
        return self._optim

    def step(self):
        return self._optim.step()
    
    ### Original
    def look_backward(self, loss):
        grads, shapes, has_grads = self._pack_grad(loss)
        #grad = self._unflatten_grad(grads, shapes)
        #self._set_grad(grad)
        n_grads = []
        for g in grads:
            if len(g.size())==2 or len(g.size())==4:
                for i in range(g.size(0)):
                    n_grads.append(g[i].norm())
        return n_grads
      
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