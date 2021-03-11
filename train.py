
import time
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from learning_rate import adjust_learning_rate
from six.moves import urllib
def train(log_interval, model, device, train_loader, optimizer, epoch,parameter_list):
    tik=time.time()
    model.train() #train모드로 설정
    running_loss =0.0
    correct=0
    num_training_data=len(train_loader.dataset)
    criterion = nn.CrossEntropyLoss() #defalut is mean of mini-batchsamples, loss type설정
    # loss함수에 softmax 함수가 포함되어있음
    for batch_idx, (data, target) in enumerate(train_loader): # 몇개씩(batch size) 로더에서 가져올지 정함 #enumerate로 batch_idx표현
        data, target = data.to(device), target.to(device) #gpu로 올림
        optimizer.zero_grad()# optimizer zero로 초기화
        output = model(data) #model에서 입력과 출력이 나옴 batch 수만큼 들어가서 batch수만큼 결과가 나옴 (1개 인풋 1개 아웃풋 아님)
        loss = criterion(output, target) #결과와 target을 비교하여 계산 

        loss.backward() #역전파
        optimizer.step() # step
        p_groups=optimizer.param_groups # group에 각 layer별 파라미터
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        parameter_list.append([])
        for p in p_groups:
            for p_layers in p['params']:
                parameter_list[-1].append(p_layers.view(-1).detach().cpu().clone()) #save cpu
                p_layers.to(device)# gpu
        
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), num_training_data,100. * batch_idx / len(train_loader), loss.item()),end='')
        
    running_loss/=num_training_data
    tok=time.time()
    running_accuracy=100. * correct / float(num_training_data)
    print('\nTrain Loss: {:.6f}'.format(running_loss),'Learning Time: {:.1f}s'.format(tok-tik),'Accuracy: {}/{} ({:.2f}%)'.format(correct, num_training_data,100.*correct/num_training_data))
    return running_accuracy,running_loss,parameter_list

def eval(model, device, test_loader,config):
    model.eval()
    eval_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum') #add all samples in a mini-batch
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            eval_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss =eval_loss/ len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        eval_loss, correct, len(test_loader.dataset),
        100. * correct / float(len(test_loader.dataset)) ) )
    eval_accuracy=100.*correct/float(len(test_loader.dataset))

    return eval_accuracy,eval_loss

def extract_data(config,time_data):
    print("Training")
    DEVICE=config['device']
    current_path = os.path.dirname(os.path.abspath(__file__))
    log_interval=100

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    from DataSet.load import data_loader
    train_data_loader,test_data_loader=data_loader(config)
    print(train_data_loader.dataset.train_data.size())
    print(test_data_loader.dataset.train_data.size())

    if config['nn_type']=='lenet5':
        from NeuralNet.lenet5 import LeNet5
        net = LeNet5().to(DEVICE)

    # Tensorboard
    logWriter=SummaryWriter(os.path.join(current_path,'training_data',time_data))
    
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    parameter_list=list()
    eval_accuracy,eval_loss=0.0,0.0
    train_accuracy,train_loss=0.0,0.0
    for epoch in range(1, config['epochs'] + 1):
        adjust_learning_rate(optimizer, epoch,config)
        train_accuracy,train_loss,parameter_list=train(log_interval,net,DEVICE,train_data_loader,optimizer,epoch,parameter_list)
        eval_accuracy,eval_loss=eval(net,DEVICE,test_data_loader,config)
        loss_dict={'train':train_loss,'eval':eval_loss}
        accuracy_dict={'train':train_accuracy,'eval':eval_accuracy}
        logWriter.add_scalars('loss',loss_dict,epoch)
        logWriter.add_scalars('accuracy',accuracy_dict,epoch)
        if config['device']=='gpu':
            torch.cuda.empty_cache()

    
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
    param_size=list()
    params_write=list()
    if config['colab']==True:
        making_path=os.path.join('drive','MyDrive','grad_data')
    else:
        making_path=os.path.join(current_path,'grad_data')
    if os.path.exists(making_path) == False:
        os.mkdir(making_path)
    tik=time.time()
    for t,params in enumerate(parameter_list):
        if t==1:
            for i, p in enumerate(params):# 각 layer의 params
                param_size.append(p.size())
        params_write.append(torch.cat(params,dim=0).unsqueeze(0))

        
        if t % 100 == 0:
            print("\r step {} done".format(t),end='')
    write_data=torch.cat(params_write,dim=0)
    print("\n Write data size:",write_data.size())
    data_frame=pd.DataFrame(write_data.numpy(),)
    data_frame.to_csv(os.path.join(making_path,'grad_{}.csv').format(time_data),index=False,header=False)
    tok=time.time()
    print('play_time for saving:',tok-tik,"s")
    print('parameter size',param_size)
    print('# of row:',t+1)

    '''
    Save params
    '''
    return config



