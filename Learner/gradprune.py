import time
import os
import sys

import torch
from Learner.base_learner import BaseLearner
from Pruning.lookupgrad import LookUpGrad
from Pruning.LRP import LateralInhibition

class GradPruneLearner(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(GradPruneLearner,self).__init__(model,time_data,file_path,configs)
        if configs['mode']=='train_grad_visual':
            self.optimizer=LookUpGrad(optimizer=self.optimizer)
        elif configs['mode']=='train_lrp':
            self.optimizer=LateralInhibition(optimizer=self.optimizer)
        self.class_idx=1
        if os.path.exists(os.path.join(self.making_path,time_data)) == False:
            os.mkdir(os.path.join(self.making_path,time_data))

    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))

        best_accuracy=0.0
        # Train
        for epoch in range(self.configs['start_epoch'], self.configs['epochs'] + 1):
                
            print('Learning rate: {}'.format(self.scheduler.optimizer.param_groups[0]['lr']))
            train_metric = self._train(epoch)
            eval_metric = self._eval()

            self.scheduler.step()
            loss_dict = {'train': train_metric['loss'], 'eval': eval_metric['loss']}
            accuracy_dict = {'train': train_metric['accuracy'], 'eval': eval_metric['accuracy']}
            self.logWriter.add_scalars('loss', loss_dict, epoch)
            self.logWriter.add_scalars('accuracy', accuracy_dict, epoch)
            best_accuracy=max(eval_metric['accuracy'],best_accuracy)

            self.early_stopping(eval_metric['loss'], self.model)
            if isinstance(self.optimizer,LookUpGrad):
                if epoch==1:
                    grads_pool=train_metric['batch_grad'].detach().clone()
                else:
                    grads_pool=torch.cat((grads_pool,train_metric['batch_grad']),dim=0)


            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            if isinstance(self.optimizer,LookUpGrad):
                torch.save(grads_pool,os.path.join(self.making_path,self.time_data,'{}-class_grads.pth.tar'.format(self.class_idx)))
        print("Best Accuracy: "+str(best_accuracy))
        self.configs['train_end_epoch']=epoch
        configs = self.save_grad(epoch)
        return configs

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
            
            if isinstance(self.optimizer,LookUpGrad):
                batch_n_grad=self.optimizer.look_backward(loss)
                if batch_idx==0:
                    batch_grads=torch.tensor(batch_n_grad,device=self.device).unsqueeze(0)
                else:
                    batch_grads=torch.cat([batch_grads,torch.tensor(batch_n_grad,device=self.device).unsqueeze(0)],dim=0)
            elif isinstance(self.optimizer,LateralInhibition):
                self.optimizer.backward(loss)

                p_groups = self.optimizer.param_groups  # group에 각 layer별 파라미터
                self.grad_list.append([])
                # grad save(prune후 save)
                self._save_grad(p_groups, epoch, batch_idx)
                
            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.item()), end='')
            if self.configs['log_extraction']=='true':
                sys.stdout.flush()

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        train_metric={'accuracy':running_accuracy,'loss': running_loss}
        if isinstance(self.optimizer,LookUpGrad):
            train_metric['batch_grad']=batch_grads
        return train_metric

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
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))
        eval_metric={'accuracy':eval_accuracy,'loss': eval_loss}
        return eval_metric
