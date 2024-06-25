import argparse
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset.dataset_forecasting import read_data2, DatasetETT
from Dataset.dataset_classification import read_data

parser = argparse.ArgumentParser(description='STMixup')
#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='forecasting',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='ETTh1')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--batchsize',type=int,default=512)
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--seqlen',type=int,default=96)
parser.add_argument('--outlen',type=int,default=96)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--dmodel',type=int,default=64)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='None',
                    choices=['Raw','Trend','Season','Final','None'])
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--tcn_kernel',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--rate',type=float,default=0.8)
parser.add_argument('--univariate',type=bool,default=True)
parser.add_argument('--chosen_list',type=int,default=-1)
parser.add_argument('--checkpoint_num',type=int,default=5)
parser.add_argument('--seed', type=int, nargs='+', default=[42,43])
parser.add_argument('--gpu',type=str,default='cuda:0')
parser.add_argument('--scale',type=str,default='minmax',choices=['std', 'minmax'])
parser.add_argument('--test',type=bool,default=True)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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

path='Data'
filepath = os.path.join(path,args.direction+'.csv')
train_data, test_data = read_data2(filepath, rate=args.rate, univariate=args.univariate,chosen_list=args.chosen_list)
scale = 'std'
device = torch.device(args.gpu)
def mixup_process(out,lam):
    batch_size = out.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_out = lam * out + (1 - lam) * out[index, :]

    return mixed_out

def mixup_process_fft(out,target,lam,range=0.5):
    batch_size = out.size()[0]
    index = torch.randperm(batch_size).cuda()
    out_fft=torch.fft.rfft(out,dim=1)
    seq_len = out_fft.size()[1]
    phase = torch.angle(out_fft)
    mixed_amp = lam*torch.abs(out_fft[:,int(seq_len*range):,:])+(1-lam)*torch.abs(out_fft[index, int(seq_len*range):,:])
    mixed_amp = torch.cat((torch.abs(out_fft[:,:int(seq_len*range),:]),mixed_amp),dim=1)
    complex_spec = torch.complex(mixed_amp*torch.cos(phase),mixed_amp*torch.sin(phase))
    ifft_result = torch.fft.irfft(complex_spec,dim=1)
    mixed_y = lam*target+(1-lam)*target[index, :]
    return ifft_result,mixed_y



if scale == 'std':
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
if scale == 'minmax':
    data_min = train_data.min(axis=0)
    data_max = train_data.max(axis=0)
    train_data = (train_data - data_min) / (data_max - data_min)
    test_data = (test_data - data_min) / (data_max - data_min)



decomp = series_decomp(32)
train_dataset = DatasetETT(train_data, dim=1, input_len=args.seqlen, output_len=args.outlen, step=int(args.seqlen/2), device=device)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
with torch.no_grad():
    for x,y in train_loader:
        lam = np.random.beta(0.5, 0.5)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        x1 = torch.cat((x,y),dim=1)
        x2 = x1[index, :]
        res1,mean1 = decomp(x1)
        res2,mean2 = decomp(x2)
        new_x = lam * x1 + (1-lam) * x2
        new_res = lam * res1 + (1-lam) * res2
        new_mean = lam * mean1 + (1-lam) * mean2
        x1,x2,new_x,res1,res2,new_res,mean1,mean2,new_mean = x1.squeeze(),x2.squeeze(),new_x.squeeze(),res1.squeeze(),res2.squeeze(),new_res.squeeze(), mean1.squeeze(), mean2.squeeze(), new_mean.squeeze()

        plt.figure(1,figsize=(32,16))
        plt.subplot(2,2,1)
        plt.plot(x1[0].cpu().numpy(),label='x1',c='r')
        plt.plot(x2[0].cpu().numpy(), label='x2',c='b')
        plt.plot(new_x[0].cpu().numpy(), label='new_x', c='g')
        plt.title('Raw Data',fontsize=40)
        plt.legend(fontsize=20)
        plt.subplot(2, 2, 2)
        plt.plot(mean1[0].cpu().numpy(), label='trend1', c='r')
        plt.plot(mean2[0].cpu().numpy(), label='trend2', c='b')
        plt.plot(new_mean[0].cpu().numpy(), label='new trend', c='g')
        plt.title('Trend',fontsize=40)
        plt.legend(fontsize=20)
        plt.subplot(2, 2, 3)
        plt.plot(res1[0].cpu().numpy(), label='season1', c='r')
        plt.plot(res2[0].cpu().numpy(), label='season2', c='b')
        plt.plot(new_res[0].cpu().numpy(), label='new season', c='g')
        plt.title('Season',fontsize=40)
        plt.legend(fontsize=20)
        plt.subplot(2, 2, 4)
        plt.plot(res1[0].cpu().numpy()+mean1[0].cpu().numpy(), label='x1', c='r')
        plt.plot(res2[0].cpu().numpy()+mean2[0].cpu().numpy(), label='x2', c='b')
        plt.plot(new_res[0].cpu().numpy()+new_mean[0].cpu().numpy(), label='new_x', c='g')
        plt.title('', fontsize=40)
        plt.legend(fontsize=20)

        plt.show()
        plt.close(1)