from Dataset.dataset_classification import read_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from matplotlib import rcParams

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size-1-(self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block

    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean




plt.rcParams['font.sans-serif'] = 'SimSun'
plt.rcParams['axes.unicode_minus']=False
plt.xticks(fontproperties='Times New Roman')
plt.yticks(fontproperties='Times New Roman')

file = 'Data/all_six_datasets/powerGF09/powerGF09.csv'
df = pd.read_csv(file,encoding='gbk')
target = df.iloc[:,-1]
length = len(target)
x1 = int(0.7*length)
x2 = int(0.8*length)
plt.plot(target[:x1],label='训练集')
plt.plot(target[x1:x2],c='orange',label='验证集')
plt.plot(target[x2:],c='g',label='测试集')
plt.xlabel('时间步',fontproperties='SimSun',fontsize=13)
plt.ylabel('数值',fontproperties='SimSun',fontsize=13)
plt.title('ETTh1',fontproperties='Times New Roman',fontsize=13)
plt.legend(prop={'size': 13})
plt.show()

