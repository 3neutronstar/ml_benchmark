import numpy as np
import copy

def softmax(a) : 
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

class Linear():
    def __init__(self,input_size,output_size):
        self.output_size=output_size
        self.input_size=input_size
        self.W=np.random.randn(output_size,input_size)*0.01
        self.b=np.zeros(output_size)
        self.cache=None
    
    def __call__(self, x):
        z=np.matmul(self.W,x)+self.b
        self.cache=z
        return z
    
    def backward(self,dz):
        dx=dz
        return dx

class ReLU():
    def __init__(self):
        self.cache=None
    def __call__(self,x):
        a=max(0,x)
        self.cache=a
        return a
    def backward(self,dz):
        dx=dz
        return dx

class CrossEntropyLoss():
    def __init__(self):
        self.cache=None
    def __call__(self,x,target):
        y=np.zeros_like(x)
        y[target]=1
        self.cache=y
        logprobs=np.matmul(np.math.log(softmax(x)),y)
        loss=-np.sum(logprobs)
        loss=float(np.squeeze(loss))
        return loss

class NumpyLeNet300_100():
    def __init__(self,configs):
        self.fc1=Linear(32*32,300)
        self.a1=ReLU()
        self.fc2=Linear(300,100)
        self.a2=ReLU()
        self.fc3=Linear(100,10)
        self.sequential=[self.fc1,self.a1,self.fc2,self.a2,self.fc3]

    def forward(self,x):
        x.reshape(x.shape[0],-1)
        for layer in self.sequential:
            x=layer(x)
        return x

    def backward(self,loss):
        y=loss
        for layer in reversed(self.sequential):
            y=layer.backward(y)