import torch
import math
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from utils import load_params
from Visualization.plot_type import using_plt,using_tensorboard
def load_csv(path,file_name):
    if file_name is None:
        csvfile=open(os.path.join(path,'grad.csv'),mode='r')
    else:
        csvfile=open(os.path.join(path,'grad_{}.csv'.format(file_name)),mode='r')
    csvReader=csv.reader(csvfile)
    return csvfile,csvReader

def visualization(config,file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    if config['colab']==True:
        path=os.path.join('drive','MyDrive','grad_data')
    else:
        path=os.path.join(current_path,'grad_data')
    csvfile,csvReader=load_csv(path,file_name)
    if file_name is None:
        CALL_CONFIG=config
    else:
        CALL_CONFIG=load_params(config,file_name)

    dir_list=['layer_grad_distribution','node_info']
    for dir_name in dir_list:
        if os.path.exists(os.path.join(path,dir_name)) == False:
            os.mkdir(os.path.join(path,dir_name))
    
    
    print("Reading file complete")
    # Structure
    # time layer element
    # grad_data = weight ,bias 순서의 layer별 데이터
    # weight_data=weight의 time list, layer list, element tensor
    # data read
    if True:
        using_tensorboard(csvReader,CALL_CONFIG,path,file_name)
    else:
        using_plt(csvReader,CALL_CONFIG,path)


    csvfile.close()
        