import time
import os
import sys
import torch
from torch import optim
from Learner.base_learner import BaseLearner
from CustomOptimizer.pcgrad import *
class MOOLearner(BaseLearner):
    def __init__(self, model, time_data,file_path, configs):
        super(MOOLearner,self).__init__(model,time_data,file_path,configs)

        if os.path.exists(os.path.join(self.making_path, time_data)) == False:
            os.mkdir(os.path.join(self.making_path, time_data))
        # if os.path.exists(os.path.join(self.making_path, 'png')) == False:
        #     os.mkdir(os.path.join(self.making_path, 'png'))

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
        if 'train_moo' in self.configs['mode']:
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
        self._show_conflicting_grad(epoch)
        return train_metric

    def _eval(self):
        self.model.eval()
        eval_loss = 0
        correct = 0
        class_correct_dict=dict()
        class_total_dict=dict()
        for i in range(self.configs['num_classes']):
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

    def _show_conflicting_grad(self,epoch):
        if self.optimizer.conflict_list != None :
            if self.configs['mode']=='baseline_v2':
                for i,i_grad_conflict in enumerate(self.optimizer.conflict_list):
                    for i_j,i_j_conflict in enumerate(i_grad_conflict):
                        if epoch ==1:
                            self.logWriter.add_histogram('conflict/{}_{}_CosineSimiarity'.format(i,i_j),torch.tensor([-1.0,1.0]),0)
                        self.logWriter.add_histogram('conflict/{}_{}_CosineSimiarity'.format(i,i_j),torch.tensor(i_j_conflict),epoch)
            elif self.configs['mode']=='baseline_v3':
                for s_l,layer_conflict_list in zip(self.optimizer.searching_layer,self.optimizer.layer_conflict_list):
                    for i,i_grad_conflict in enumerate(layer_conflict_list):
                        for i_j,i_j_conflict in enumerate(i_grad_conflict):
                            if epoch ==1:
                                self.logWriter.add_histogram('{}l_conflict/{}_{}_CosineSimiarity'.format(s_l,i,i_j),torch.tensor([-1.0,1.0]),0)
                            self.logWriter.add_histogram('{}l_conflict/{}_{}_CosineSimiarity'.format(s_li,i_j),torch.tensor(i_j_conflict),epoch)
            

        self.optimizer.conflict_list=None