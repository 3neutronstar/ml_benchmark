from pandas.core.frame import DataFrame
import torch
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from utils import load_params
from Visualization.plot_type import using_plt,using_tensorboard
def load_csv(path,file_name):
    if file_name is None:
        dataFile=open(os.path.join(path,'grad.csv'),mode='r')
    else:
        csvfile=open(os.path.join(path,'grad_{}.csv'.format(file_name)),mode='r')
    csvReader=pd.read_csv(csvfile,dtype=float)
    dataTensor=torch.from_numpy(pd.get_dummies(csvReader).values)
    print("Load success csv, Size: ",dataTensor.size())
    return dataFile,dataTensor

def load_npy(path,file_name):
    if file_name is None:
        dataFile=np.load(os.path.join(path,'grad.npy'))
    else:
        dataFile=np.load(os.path.join(path,'grad_{}.npy'.format(file_name)))
    dataTensor=torch.from_numpy(dataFile)
    print("Load success npy, Size: ",dataTensor.size())
    return dataTensor

def visualization(config,file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    if config['colab']==True:
        path=os.path.join('drive','MyDrive','grad_data')
    else:
        path=os.path.join(current_path,'grad_data')
    # dataFile,dataTensor=load_csv(path,file_name)
    dataTensor=load_npy(path,file_name)
    if file_name is None:
        CALL_CONFIG=config
    else:
        CALL_CONFIG=load_params(config,file_name)
    
    tik=time.time()
    # dir_list=['layer_grad_distribution','node_info']
    # for dir_name in dir_list:
    #     if os.path.exists(os.path.join(path,dir_name)) == False:
    #         os.mkdir(os.path.join(path,dir_name))
    # Structure
    # time layer element
    # grad_data = weight ,bias 순서의 layer별 데이터
    # weight_data=weight의 time list, layer list, element tensor
    # data read
    if True:
        using_tensorboard(dataTensor,CALL_CONFIG,path,file_name)
    else:
        using_plt(dataTensor,CALL_CONFIG,path)
    
    tok=time.time()
    print("\n Visualization Time: {}s".format(tik-tok))

        