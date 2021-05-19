import torch

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class LayerByLayerOptimizer():
    def __init__(self, model,optimizer):
        self._optim = optimizer
        self._model=model
        self.hookBackward=[Hook(layer[1],backward=True) for layer in list(model._modules.items())]
        self.hookForward=[Hook(layer[1],backward=False) for layer in list(model._modules.items())]
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
        for i,(fhook,bhook) in enumerate(reversed(zip(self.hookForward,self.hookBackward))):
            if i==0:
                for pc_obj in pc_objectives:
                    self._optim.zero_grad()
                    pc_obj.backward(retain_graph=True)

        return


'''
        What is the input and output of forward and backward pass?
Things to notice:
Because backward pass runs from back to the start, it's parameter order should be reversed compared to the forward pass. Therefore, to be it clearer, I'll use a different naming convention below.
For forward pass, previous layer of layer 2 is layer1; for backward pass, previous layer of layer 2 is layer 3.
Model output is the output of last layer in forward pass.
layer.register_backward_hook(module, input, output)

Input: previous layer's output
Output: current layer's output
layer.register_backward_hook(module, grad_out, grad_in)

Grad_in: gradient of model output wrt. layer output       # from forward pass
= a tensor that represent the error of each neuron in this layer (= gradient of model output wrt. layer output = how much it should be improved)
For the last layer: eg. [1,1] <=> gradient of model output wrt. itself, which means calculate all gradients as normal
It can also be considered as a weight map: eg. [1,0] turn off the second gradient; [2,1] put double weight on first gradient etc.

Grad_out: Grad_in * (gradient of layer output wrt. layer input)
= next layer's error(due to chain rule)
Check the print from the cell above to confirm and enhance your understanding!
'''

'''
Modify gradients with hooks
Hook function doesn't change gradients by default
But if return is called, the returned value will be the gradient output
'''