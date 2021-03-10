
import torch
import torch.nn as nn
import torch.optim as optim
from learning_rate import adjust_learning_rate
from six.moves import urllib
import time
def train(log_interval, model, device, train_loader, optimizer, epoch,parameter_list):
    tik=time.time()
    model.train() #train모드로 설정
    running_loss =0.0
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
        parameter_list.append([])
        for p in p_groups:
            for p_layers in p['params']:
                parameter_list[-1].append(p_layers.view(-1).detach().cpu().clone())
        
        running_loss += loss.cpu().item()
        if batch_idx % log_interval == 0:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), running_loss/log_interval),end='')
    tok=time.time()
    print('Train Loss: {:.6f}'.format( running_loss/len(data)),'Learning Time: ',tok-tik,'s')
    return running_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum') #add all samples in a mini-batch
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / float(len(test_loader.dataset)) ) )
    accuracy=100.*correct/float(len(test_loader.dataset))
    return accuracy

def extract_data(configs):
    print("Training")
    DEVICE=configs['device']
    log_interval=100

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    from DataSet.load import data_loader
    train_data_loader,test_data_loader=data_loader(configs)

    
    print(train_data_loader.dataset.train_data.size())
    print(test_data_loader.dataset.train_data.size())

    if configs['nn_type']=='lenet5':
        from NeuralNet.lenet5 import LeNet5
        net = LeNet5().to(DEVICE)
    
    optimizer = optim.Adam(net.parameters(), lr=configs['lr'])
    parameter_list=list()
    final_accuracy=0.0
    final_loss=0.0
    for epoch in range(1, configs['epochs'] + 1):
        adjust_learning_rate(optimizer, epoch,configs)
        final_loss=train(log_interval,net,DEVICE,train_data_loader,optimizer,epoch,parameter_list)
        final_accuracy=test(net,DEVICE,test_data_loader)
        if configs['device']=='gpu':
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
    import time
    import csv
    import os
    param_size=list()
    params_write=list()
    current_path = os.path.dirname(os.path.abspath(__file__))
    if configs['colab']==True:
        making_path=os.path.join('drive','MyDrive','grad_data')
    else:
        making_path=os.path.join(current_path,'grad_data')
    if os.path.exists(making_path) == False:
        os.mkdir(making_path)
    f=open(os.path.join(making_path,'grad.csv'),mode='w')
    fwriter=csv.writer(f)
    tik=time.time()
    for t,params in enumerate(parameter_list):
        if t==1:
            for i, p in enumerate(params):# 각 layer의 params
                param_size.append(p.size())
        params_write=torch.cat(params,dim=0).tolist()
        fwriter.writerow(params_write)
        if t % 100 == 0:
            print("\r step {} done".format(t),end='')
    f.close()
    tok=time.time()
    print('play_time for saving:',tok-tik,"s")
    print('parameter size',param_size)
    print('# of row:',t+1)

    '''
    Save params
    '''
    configs['final_accuracy']=final_accuracy
    configs['final_loss']=final_loss

    return configs



