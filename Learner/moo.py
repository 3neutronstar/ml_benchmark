import time
import os
import sys
import torch
from torch import optim
from Learner.base_learner import BaseLearner
from CustomOptimizer.pcgrad import PCGrad_MOO,PCGrad_MOO_Baseline,PCGrad_MOO_V2,PCGrad_MOO_Baseline_V2
class MOOLearner(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(MOOLearner,self).__init__(model,time_data,file_path,configs)
        if 'train_moo' == configs['mode']:
            reduction='none'
            self.optimizer=PCGrad_MOO(self.optimizer)
        elif 'train_moo_v2' == configs['mode']:
            reduction='none'
            self.optimizer=PCGrad_MOO_V2(self.optimizer)
        elif 'baseline_moo' == configs['mode']:
            reduction='mean'
            self.optimizer=PCGrad_MOO_Baseline(self.optimizer)
        elif 'baseline_moo_v2' == configs['mode']:
            reduction='none'
            self.optimizer=PCGrad_MOO_Baseline_V2(self.optimizer)
        else:
            raise NotImplementedError

        self.criterion=self.criterion.__class__(reduction=reduction) # grad vector (no scalar)
        if os.path.exists(os.path.join(self.making_path, time_data)) == False:
            os.mkdir(os.path.join(self.making_path, time_data))

    def run(self):
        print("Training {} epochs".format(self.configs['epochs']))
        best_accuracy=0.0
        # Train
        for epoch in range(1, self.configs['epochs'] + 1):
                
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
            if self.device == 'gpu':
                torch.cuda.empty_cache()
        if 'train' in self.configs['mode']:
            print("Total Conflict Number: {}".format(self.optimizer.total_conflict_num))
        print("Best Accuracy: "+str(best_accuracy))
        self.configs['train_end_epoch']=epoch
        configs = self.save_grad(epoch)
        return configs


    def _train(self, epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        total_len_data=0
        len_data=dict()
        class_correct_dict=dict()
        for i in range(self.configs['num_classes']):
            class_correct_dict[i]=0
            len_data[i]=0
        current_len_data=0
        total_len_data=len(self.train_loader.dataset)
        for idx,(data,target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)  # gpu로 올림
            output = self.model(data)
            loss = self.criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            for class_idx in target.unique():
                class_correct_dict[int(class_idx)]+=pred.eq(target.view_as(pred))[target==class_idx].sum().item()
                len_data[int(class_idx)]+=(target==class_idx).sum()

            running_loss+=loss.mean().item()
            self.optimizer.pc_backward(loss,target,epoch)
            self.optimizer.step()

            current_len_data+=target.size()[0]
            if idx % self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, current_len_data, total_len_data ,
                                                                                100.0 * float(current_len_data) / float(total_len_data), loss.mean().item()), end='')

        tok=time.time()
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        print("\n ============================\nTrain Learning Time:{:.2f}s \t Class Accuracy".format(tok-tik))
        total_correct=0
        for class_correct_key in class_correct_dict.keys():
            class_accur=100.0*float(class_correct_dict[class_correct_key])/float(len_data[class_correct_key])
            print('{} class :{}/{} {:2f}%'.format(class_correct_key,class_correct_dict[class_correct_key],len_data[class_correct_key],class_accur))
            total_correct+=class_correct_dict[class_correct_key]
        running_accuracy=100.0*float(total_correct)/float(total_len_data)
        train_metric={'accuracy':running_accuracy,'loss': running_loss/float(total_len_data)}
        print('{} epoch Total Accuracy: {:.2f}%, Total Loss: {}\n'.format(epoch,train_metric['accuracy'],train_metric['loss']))
        if 'train' in self.configs['mode']:
            print("The number of conflict Number:",self.optimizer.epoch_conflict_num)
            self.optimizer.epoch_conflict_num=list()
                
        return train_metric

    def _eval(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        class_correct_dict=dict()
        class_total_dict=dict()
        for i in range(self.configs['moo_num_classes']):
            class_correct_dict[i]=0
            class_total_dict[i]=0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                eval_loss += loss.mean().item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                for label in target.unique():
                    # print(label,pred.eq(target.view_as(pred))[target==label].sum().item())
                    class_correct_dict[int(label)]+=pred.eq(target.view_as(pred))[target==label].sum().item()
                    class_total_dict[int(label)]+=(target==label).sum().item()

        eval_loss = eval_loss / len(self.test_loader.dataset)

        correct=0
        print("=================Eval=================")
        for class_correct_key in class_correct_dict.keys():
            correct+=class_correct_dict[class_correct_key]
            class_accur=100.0*float(class_correct_dict[class_correct_key])/class_total_dict[class_correct_key]
            print('{} class :{}/{} {:2f}%'.format(class_correct_key,class_correct_dict[class_correct_key],class_total_dict[class_correct_key],class_accur))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n==================='.format(
            eval_loss, correct, len(self.test_loader.dataset),
            100.0 * correct / float(len(self.test_loader.dataset))))
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        eval_accuracy = 100.0*correct/float(len(self.test_loader.dataset))
        eval_metric={'accuracy':eval_accuracy,'loss': eval_loss}

        return eval_metric

        