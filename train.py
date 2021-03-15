
import time
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from six.moves import urllib
from torch.optim. lr_scheduler import StepLR, MultiStepLR
from utils import EarlyStopping


def train(log_interval, model, device, train_loader, optimizer, epoch, parameter_list,config):
    tik = time.time()
    model.train()  # train모드로 설정
    running_loss = 0.0
    correct = 0
    num_training_data = len(train_loader.dataset)
    # defalut is mean of mini-batchsamples, loss type설정
    criterion = model.loss
    # loss함수에 softmax 함수가 포함되어있음
    # 몇개씩(batch size) 로더에서 가져올지 정함 #enumerate로 batch_idx표현
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # gpu로 올림
        optimizer.zero_grad()  # optimizer zero로 초기화
        # model에서 입력과 출력이 나옴 batch 수만큼 들어가서 batch수만큼 결과가 나옴 (1개 인풋 1개 아웃풋 아님)
        output = model(data)
        loss = criterion(output, target)  # 결과와 target을 비교하여 계산

        loss.backward()  # 역전파
        optimizer.step()  # step
        p_groups = optimizer.param_groups  # group에 각 layer별 파라미터
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        parameter_list.append([])
        for p in p_groups:
            for i,p_layers in enumerate(p['params']):
                # save cpu
                # filtering the info by norm, avg, var
                # if i%2==0:#==0:weight// bias filtering
                #     p_calc=p_layers.view(p_layers.size()[0],-1).cpu().detach().clone()
                #     # if i==0:
                #     #     print(p_calc[0])
                #     parameter_list[-1].append(torch.cat([p_calc.mean(dim=1,keepdim=True),p_calc.norm(dim=1,keepdim=True),p_calc.var(dim=1,keepdim=True)],dim=1))
                if i%2==0:
                    p_node=p_layers.view(p_layers.size()[0],-1).cpu().detach().clone()
                    for p_elem in p_node:
                        parameter_list[-1].append(p_elem)
                    p_layers.to(device)  # gpu

        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(
                data), num_training_data, 100.0 * batch_idx / len(train_loader), loss.item()), end='')

    running_loss /= num_training_data
    tok = time.time()
    running_accuracy = 100.0 * correct / float(num_training_data)
    print('\nTrain Loss: {:.6f}'.format(running_loss), 'Learning Time: {:.1f}s'.format(
        tok-tik), 'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data, 100.0*correct/num_training_data))
    return running_accuracy, running_loss, parameter_list


def eval(model, device, test_loader, config):
    model.eval()
    eval_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum')  # add all samples in a mini-batch
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            eval_loss += loss.item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss = eval_loss / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        eval_loss, correct, len(test_loader.dataset),
        100.0 * correct / float(len(test_loader.dataset))))
    eval_accuracy = 100.0*correct/float(len(test_loader.dataset))

    return eval_accuracy, eval_loss


def extract_data(config, time_data):
    print("Training {} epochs".format(config['epochs']))
    DEVICE = config['device']
    current_path = os.path.dirname(os.path.abspath(__file__))
    log_interval = 100

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    from DataSet.data_load import data_loader
    train_data_loader, test_data_loader = data_loader(config)

    if config['nn_type'] == 'lenet5':
        from NeuralNet.lenet5 import LeNet5
        net = LeNet5(config).to(DEVICE)
    if config['nn_type'][:3] == 'vgg':
        from NeuralNet.vgg import VGG
        net = VGG(config).to(DEVICE)

    # Tensorboard
    logWriter = SummaryWriter(os.path.join(
        current_path, 'training_data', time_data))

    optimizer = net.optim

    if config['nn_type'] == 'lenet5':
        scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1)
    elif config['nn_type'][:3] == 'vgg':
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                150, 225], gamma=0.1)

    parameter_list = list()
    eval_accuracy, eval_loss = 0.0, 0.0
    train_accuracy, train_loss = 0.0, 0.0
    early_stopping = EarlyStopping(current_path,time_data,patience = config['patience'], verbose = True)
    # Train
    for epoch in range(1, config['epochs'] + 1):
        train_accuracy, train_loss, parameter_list = train(
            log_interval, net, DEVICE, train_data_loader, optimizer, epoch, parameter_list,config)
        eval_accuracy, eval_loss = eval(net, DEVICE, test_data_loader, config)
        scheduler.step()
        loss_dict = {'train': train_loss, 'eval': eval_loss}
        accuracy_dict = {'train': train_accuracy, 'eval': eval_accuracy}
        logWriter.add_scalars('loss', loss_dict, epoch)
        logWriter.add_scalars('accuracy', accuracy_dict, epoch)

        early_stopping(eval_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        if config['device'] == 'gpu':
            torch.cuda.empty_cache()
    # Save
    param_size = list()
    params_write = list()
    if config['colab'] == True:
        making_path = os.path.join('drive', 'MyDrive', 'grad_data')
    else:
        making_path = os.path.join(current_path, 'grad_data')
    if os.path.exists(making_path) == False:
        os.mkdir(making_path)
    tik = time.time()
    import numpy as np
    if config['log_extraction'] == True:
        for t, params in enumerate(parameter_list):
            if t == 1:
                for i, p in enumerate(params):  # 각 layer의 params
                    param_size.append(p.size())
            
            params_write.append(torch.cat(params,dim=0).unsqueeze(0))

            if t % 100 == 0:
                print("\r step {} done".format(t), end='')
        write_data = torch.cat(params_write, dim=0)
        print("\n Write data size:", write_data.size())
        np.save(os.path.join(making_path, 'grad_{}'.format(
            time_data)), write_data.numpy())#npy save
        tok = time.time()
        print('play_time for saving:', tok-tik, "s")
        print('size: {}'.format(len(params_write)))
    #CSV extraction
    '''
    csv 저장
    자료구조
    parameter_list
    1dim: time
    2dim: layer

    저장시
    x 축 내용: parameters의 grad
    x 축 : time
    y 축 : param
    '''
    # tik=time.time()
    # params_write.clear()
    # if config['csv_extraction']==True:
    #     for t,params in enumerate(parameter_list):
    #         if t==1:
    #             for i, p in enumerate(params):# 각 layer의 params
    #                 param_size.append(p.size())
    #         params_write.append(torch.cat(params,dim=0).unsqueeze(0))

    #         if t % 100 == 0:
    #             print("\r step {} done".format(t),end='')
    #     write_data=torch.cat(params_write,dim=0)
    #     print("\n Write data size:",write_data.size())
    #     data_frame=pd.DataFrame(write_data.numpy(),)
    #     data_frame.to_csv(os.path.join(making_path,'grad_{}.csv').format(time_data),index=False,header=False)
    #     tok=time.time()
    #     print('play_time for saving:',tok-tik,"s")
    #     print('parameter size',param_size)
    #     print('# of row:',t+1)
    # else:
    #     print("No Extraction of csv file")

    '''
    Save params
    '''
    return config
