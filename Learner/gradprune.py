import torch
import time
import sys
from Learner.base_learner import BaseLearner
import os
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

class GradPruneLearner(BaseLearner):
    def __init__(self, model, time_data, configs):
        super(GradPruneLearner,self).__init__(model,time_data,configs)
        self.optimizer=LookUpGrad(optimizer=self.optimizer)
        self.class_idx=1
        if os.path.exists(os.path.join(self.making_path,time_data)) == False:
            os.mkdir(os.path.join(self.making_path,time_data))

    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))

        eval_accuracy, eval_loss = 0.0, 0.0
        train_accuracy, train_loss = 0.0, 0.0
        best_accuracy=0.0
        # Train
        for epoch in range(1, self.configs['epochs'] + 1):
                
            print('Learning rate: {}'.format(self.scheduler.optimizer.param_groups[0]['lr']))
            train_accuracy, train_loss,b_grad = self._train(epoch)
            eval_accuracy, eval_loss = self._eval()

            self.scheduler.step()
            loss_dict = {'train': train_loss, 'eval': eval_loss}
            accuracy_dict = {'train': train_accuracy, 'eval': eval_accuracy}
            self.logWriter.add_scalars('loss', loss_dict, epoch)
            self.logWriter.add_scalars('accuracy', accuracy_dict, epoch)
            best_accuracy=max(eval_accuracy,best_accuracy)

            self.early_stopping(eval_loss, self.model)
            if epoch==1:
                grads_pool=b_grad.detach().clone()
            else:
                grads_pool=torch.cat((grads_pool,b_grad),dim=0)
            

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if self.device == 'gpu':
                torch.cuda.empty_cache()
            torch.save(grads_pool,os.path.join(self.making_path,self.time_data,'{}-class_grads.pth.tar'.format(self.class_idx)))
        print("Best Accuracy: "+str(best_accuracy))
        self.configs['train_end_epoch']=epoch
        return self.configs

    def _train(self, epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        correct = 0
        num_training_data = len(self.train_loader.dataset)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(
                self.device)  # gpu로 올림
            output = self.model(data)
            loss = self.criterion(output, target) 

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            batch_n_grad=self.optimizer.look_backward(loss)
            if batch_idx==0:
                batch_grads=torch.tensor(batch_n_grad,device=self.device).unsqueeze(0)
            else:
                batch_grads=torch.cat([batch_grads,torch.tensor(batch_n_grad,device=self.device).unsqueeze(0)],dim=0)

            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.item()), end='')

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        return running_accuracy, running_loss,batch_grads

    def _eval(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        criterion = self.model.loss  # add all samples in a mini-batch
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                eval_loss += loss.item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss = eval_loss / len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))

        return eval_accuracy, eval_loss
