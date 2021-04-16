import time
import os
import sys
import torch
from torch import optim
from Learner.base_learner import BaseLearner
from CustomLoss.pcgrad import PCGrad, PCGrad_v3

class MTLLearner(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(MTLLearner,self).__init__(model,time_data,file_path,configs)
        if configs['mode']=='traim_mtl':
            self.optimizer=PCGrad(self.optimizer)
        elif configs['mode']=='train_mtl_v3':
            self.optimizer=PCGrad_v3(self.optimizer)
        self.class_idx=1
        self.criterion=self.criterion.__class__(reduction='none')#grad vector (no scalar)
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


            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if self.device == 'cuda':
                torch.cuda.empty_cache()
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

            self.optimizer.pc_backward(loss,target)
            self.optimizer.step()

            running_loss += loss.sum().item()
            if batch_idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                    data), num_training_data, 100.0 * batch_idx / len(self.train_loader), loss.sum().item()), end='')

        running_loss /= num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        train_metric={'accuracy':running_accuracy,'loss': running_loss}
        return train_metric

    def _eval(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                eval_loss += loss.sum().item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss = eval_loss / len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))
        eval_metric={'accuracy':eval_accuracy,'loss': eval_loss}
        return eval_metric
