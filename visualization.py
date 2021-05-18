from pandas.core.frame import DataFrame
import torch
import time
import pandas as pd
import os
import numpy as np
from Visualization.plot_type import using_plt, using_tensorboard


def load_csv(path, file_name):
    if file_name is None:
        dataFile = open(os.path.join(path, 'grad.csv'), mode='r')
    else:
        csvfile = open(os.path.join(
            path, 'grad_{}.csv'.format(file_name)), mode='r')
    csvReader = pd.read_csv(csvfile, dtype=float)
    dataTensor = torch.from_numpy(pd.get_dummies(csvReader).values)
    print("Load success csv, Size: ", dataTensor.size())
    return dataFile, dataTensor


def load_npy(path, file_name):
    if file_name is None:
        dataFile = np.load(os.path.join(path, 'grad.npy'))
    else:
        dataFile = np.load(os.path.join(path, 'grad_{}.npy'.format(file_name)))
    dataTensor = torch.from_numpy(dataFile)
    print("Load success npy, Size: ", dataTensor.size())
    return dataTensor


def visualization(config, file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    tik = time.time()
    if config['colab'] == True:
        path = os.path.join('drive', 'MyDrive', 'grad_data')
    else:
        path = os.path.join(current_path, 'grad_data')

    if True:
        using_tensorboard(config, path, file_name)
    else:
        dataTensor = load_npy(path, file_name)
        using_plt(dataTensor, config, path,file_name)

    tok = time.time()
    print("\n Visualization Time: {}s".format(tok-tik))
