import time
import os
import sys
import torch
from Learner.base_learner import BaseLearner
from CustomLoss.pcgrad import PCGrad_v2,PCGrad_v4

class MTLLearner_v2(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(MTLLearner_v2,self).__init__(model,time_data,file_path,configs)
        if configs['mode']=='train_mtl_v2':
            self.optimizer=PCGrad_v2(self.optimizer)
        if configs['mode']=='train_mtl_v4':
            self.optimizer=PCGrad_v4(self.optimizer)
        else:
            raise NotImplementedError
        self.class_idx=1
        self.criterion=self.criterion.__class__(reduction='mean')#grad vector (no scalar)
        if os.path.exists(os.path.join(self.making_path,time_data)) == False:
            os.mkdir(os.path.join(self.making_path,time_data))
        self.num_training_data=0
        self.training_data_len_list=list()
        for train_loader in self.train_loader:
            self.num_training_data += len(train_loader.dataset)
            self.training_data_len_list.append(len(train_loader))

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

    def _class_wise_write(self,data,target):
        data,target = data.to(self.device), target.to(
                self.device)
        output = self.model(data)
        loss = self.criterion(output, target) 

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        self.optimizer.pc_backward(loss,target)
        del data,target
        return correct,loss


    def _train(self, epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        correct = 0
        train_loader=list()

        for i,data_loader in enumerate(self.train_loader):
            train_loader.append(enumerate(data_loader))
        batch_idx=0
        while True:
            for i,loader in enumerate(train_loader):
                if (batch_idx+1)>=self.training_data_len_list[i]: # data갯수가 달라 batch 수가 다를경우
                    continue
                batch_idx,(data,target)=next(loader)
                _correct,loss=self._class_wise_write(data,target)
                correct+=_correct
                running_loss += loss.item()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            if batch_idx % self.log_interval == 0: 
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * self.configs['batch_size']*len(self.train_loader),
                    self.num_training_data, 100.0 * float((batch_idx+1) * self.configs['batch_size']*len(self.train_loader)) /float(self.num_training_data), loss.item()), end='')
            self.optimizer.step()

            if max(self.training_data_len_list)==(batch_idx+1):# 끝내기용
                break

        running_loss /= self.num_training_data
        tok = time.time()
        running_accuracy = 100.0 * correct / float(self.num_training_data)
        print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
            tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, self.num_training_data, 100.0*correct/self.num_training_data))
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        train_metric={'accuracy':running_accuracy,'loss': running_loss}
        del train_loader
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
                eval_loss += loss.item()
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
