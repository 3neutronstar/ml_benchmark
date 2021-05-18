import torch

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

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self,tensor_value):
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


    def _pack_grad(self, objectives,idx_reversed):
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
            grad, shape= self._retrieve_grad(self._model.input[idx_reversed+1])
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        return grads, shapes
        
    def backward(self, objectives,labels):
        '''
        calculate the gradient of the parameters over classes
        input:
        - objectives: a list of objectives
        '''
        self._optim.zero_grad()
        classwise_objectives=list()
        for label in labels.unique():
           classwise_objectives.append(objectives[labels==label].mean().view(1))
        classwise_objectives=torch.cat(classwise_objectives,dim=0)
        for i, output in reversed(list(enumerate(self._model.output))):
            if i == (len(self._model.output) - 1):
                # for last node, use g
                objectives.backward()
            else:
                output.backward(self._model.input[i+1].grad.data)
                # print(i, self._model.input[i+1].grad.data)
        return