from Learner.moo import MOOLearner
import torch
import time
import sys

def split_class_list_data_loader(train_dataloader,configs):
    import random
    train_data=train_dataloader.dataset
    
    if configs['device'] == 'gpu':
        pin_memory = True
    else:
        pin_memory = False
    
    data_classes = [i for i in range(configs['moo_num_classes'])]
    random.shuffle(data_classes)
    sparse_data_classes=data_classes[:configs['moo_num_sparse_classes']]
    data_classes.sort()
    sparse_data_classes.sort()
    train_data_loader=list()
    train_subset_dict=dict()
    for i in data_classes:
        train_subset_dict[i]=list()
    #train
    for idx,(train_images, train_label) in enumerate(train_data):
        if train_label in data_classes:
            train_subset_dict[train_label].append(idx)
        else:
            continue
    min_data_num=min([len(train_subset_dict[i]) for i in data_classes])
    # train data sparsity generator
    for i in data_classes:
        #resize batch size
        if i in sparse_data_classes:
            batch_size=int(configs['batch_size']*configs['moo_sparse_ratio']/configs['num_classes'])
        if 'train_moo' in configs['mode']:
            if i in sparse_data_classes:
                batch_size=int(configs['batch_size']*configs['moo_sparse_ratio']/configs['num_classes'])
            else:
                batch_size=int(configs['batch_size']/configs['num_classes'])
        elif configs['mode']=='baseline_moo':
            batch_size=int(configs['batch_size']/configs['num_classes'])
        else:
            batch_size=int(configs['batch_size'])
            raise NotImplementedError

        # sparse는 줄이기
        if i in sparse_data_classes:
            train_subset_dict[i]=train_subset_dict[i][:int(min_data_num*configs['moo_sparse_ratio'])]
        else:
            train_subset_dict[i]=train_subset_dict[i][:int(min_data_num)]
        # loader에 담기
        locals()['trainset_{}'.format(i)] = torch.utils.data.Subset(train_data,
                                                train_subset_dict[i]) # 인덱스 기반 subset 생성
        train_data_loader.append(torch.utils.data.DataLoader(locals()['trainset_{}'.format(i)],
                                                    batch_size=batch_size,
                                                    pin_memory=pin_memory,
                                                    shuffle=True
                                                    )) # 각 loader에 넣기
        print('{} class have {} data'.format(i,len(train_subset_dict[i])))

    print("Finish Load splitted dataset")
    return train_data_loader




class MOO_V2Learner(MOOLearner):
    def __init__(self, model, time_data, file_path, configs):
        super().__init__(model, time_data, file_path, configs)

        self.train_loader=split_class_list_data_loader(self.train_loader,configs)

    def _catenate_class_load_data(self,loaders):
        cat_data,cat_target=[],[]
        for loader in loaders:
            data,target=next(loader)
            cat_data.append(data)
            cat_target.append(target)
        cat_data=torch.cat(cat_data,dim=0)
        cat_target=torch.cat(cat_target,dim=0)

        return cat_data,cat_target


    def _train(self, epoch):
        tik = time.time()
        self.model.train()  # train모드로 설정
        running_loss = 0.0
        total_len_data=0
        len_data=dict()
        class_correct_dict=dict()
        train_loader=list()
        for i,class_data_loader in enumerate(self.train_loader):
            class_correct_dict[i]=0
            len_data[i]=len(class_data_loader.dataset)
            total_len_data+=len_data[i]
            train_loader.append(iter(class_data_loader))
        current_len_data=0
        while True:
            data, target=self._catenate_class_load_data(train_loader)
            data, target = data.to(self.device), target.to(self.device)  # gpu로 올림
            output = self.model(data)
            loss = self.criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            for class_idx in target.unique():
                class_correct_dict[int(class_idx)]+=pred.eq(target.view_as(pred))[target==class_idx].sum().item()

            self.optimizer.pc_backward(loss,target,epoch)
            running_loss+=loss.sum().item()
            self.optimizer.pc_backward(loss,target,epoch)
            self.optimizer.step()

            current_len_data+=target.size()[0]
            if total_len_data<=current_len_data:
                break
            if current_len_data % self.configs['batch_size']*self.log_interval == 0:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, current_len_data, total_len_data ,
                                                                                100.0 * float(current_len_data) / float(total_len_data), loss.sum().item()), end='')
        tok=time.time()
        if self.configs['log_extraction']=='true':
            sys.stdout.flush()
        print("\n ============================\n Learning Time:{}s \t Class Accuracy".format(tok-tik))
        print("\n ============================\n{}epoch Train Learning Time:{}s \t Class Accuracy".format(epoch,tok-tik))
        total_correct=0
        for class_correct_key in class_correct_dict.keys():
            class_accur=100.0*float(class_correct_dict[class_correct_key])/float(len_data[class_correct_key])
            print('{} class :{}/{} {:2f}%'.format(class_correct_key,class_correct_dict[class_correct_key],len_data[class_correct_key],class_accur))
            total_correct+=class_correct_dict[class_correct_key]
        running_accuracy=100.0*float(total_correct)/float(total_len_data)
        train_metric={'accuracy':running_accuracy,'loss': running_loss/float(total_len_data)}
        print('Total Accuracy: {:.2f}%, Total Loss: {}\n'.format(train_metric['accuracy'],train_metric['loss']))
        return train_metric