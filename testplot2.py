import os.path
import torch.cuda
from torch.utils.data import DataLoader
from Dataset.dataset_forecasting import read_data2,DatasetETT
import argparse
from ST_Encoder.forecasting_model import MICN,tcnforecast
from torch import optim
from utils import RMSELoss,EarlyStopping,make_dir,mse_loss,mae_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='STMixup')
#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='forecasting',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='powerGF09')
parser.add_argument('--lr',type=float,default=1e-2)
parser.add_argument('--batchsize',type=int,default=1024)
parser.add_argument('--epoch',type=int,default=500)
parser.add_argument('--seqlen',type=int,default=192)
parser.add_argument('--outlen',type=int,default=192)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--dmodel',type=int,default=128)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='None',
                    choices=['Raw','Trend','Season','Final','None','both'])
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--tcn_kernel',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--rate',type=float,default=0.8)
parser.add_argument('--univariate',type=bool,default=True)
parser.add_argument('--chosen_list',type=int,default=-1)
parser.add_argument('--checkpoint_num',type=int,default=5)
parser.add_argument('--seed', type=int, nargs='+', default=[42, 43, 44, 45, 46])
parser.add_argument('--gpu',type=str,default='cuda:1')
parser.add_argument('--scale',type=str,default='std',choices=['std', 'minmax'])
parser.add_argument('--test',type=bool,default=True)
parser.add_argument('--fftrate',type=float,default=0.01)
parser.add_argument('--mixfft',type=bool,default=False)
parser.add_argument('--decomp',type=int,nargs='+',default=[32])
parser.add_argument('--alpha',type=float,default=0.5)
parser.add_argument('--beta',type=float,default=0.5)
parser.add_argument('--valbatch',type=int,default=64)
parser.add_argument('--model', type=str, default='micn')
args = parser.parse_args()
print(vars(args))


seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = args.gpu if torch.cuda.is_available() else 'cpu'
path = 'Data/all_six_datasets'
filepath = os.path.join(path, args.direction)
filepath = os.path.join(filepath, args.direction + '.csv')
train_data, test_data = read_data2(filepath, rate=args.rate, univariate=args.univariate, chosen_list=args.chosen_list)
if args.univariate:
    dim = 1
else:
    dim = train_data.shape[-1]


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

model_name = 'checkpoints/forecasting/powerGF09/powerGF09_epoch500_bs1024_lr0.01_dm128_Season_decomp[32]_dmodel128/46/best_model1.pkl'

model = MICN(dec_in=dim,c_out=dim,seq_len=args.seqlen,out_len=args.outlen,tcn_kernel=args.tcn_kernel,
             d_model=args.dmodel,dropout=args.dropout,mode='TCN',part=args.part,layer_mix=args.mixlayer).to(device)


checkpoint=torch.load(model_name)
model.load_state_dict(checkpoint['model_state_dict'])
saved_epoch = checkpoint['epoch']
test_dataset = DatasetETT(test_data, input_len=args.seqlen, output_len=args.outlen, step=1, device=device,dim=dim)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)


def relative_error(predicted_sequence, true_sequence):
    """
    计算预测序列和真实序列的相对误差。

    参数：
    predicted_sequence: list，预测序列。
    true_sequence: list，真实序列。

    返回值：
    相对误差，以百分比表示。
    """
    if len(predicted_sequence) != len(true_sequence):
        raise ValueError("预测序列和真实序列的长度必须相等")

    error_sum = 0
    true_sum = 0
    for pred, true_val in zip(predicted_sequence, true_sequence):
        error_sum += abs(pred - true_val)
        true_sum += abs(true_val)

    relative_error = (error_sum / true_sum) * 100/len(predicted_sequence)
    return relative_error

def absolute_error(predicted_sequence, true_sequence):
    """
    计算预测序列和真实序列的相对误差。

    参数：
    predicted_sequence: list，预测序列。
    true_sequence: list，真实序列。

    返回值：
    相对误差，以百分比表示。
    """
    if len(predicted_sequence) != len(true_sequence):
        raise ValueError("预测序列和真实序列的长度必须相等")

    error_sum = 0
    true_sum = 0
    for pred, true_val in zip(predicted_sequence, true_sequence):
        error_sum += abs(pred - true_val)
        true_sum += abs(true_val)

    absolute_error = error_sum/len(predicted_sequence)
    return absolute_error


for x, y in test_loader:
    with torch.no_grad():
        model.eval()
        res = model(x)
        res = res.squeeze()
        y = y.squeeze()
        y = y.to('cpu')
        res = res.to('cpu')

        re_error = relative_error(res, y)
        ab_error = absolute_error(res, y)
        print('relative_error', re_error.item())
        print('absolute_error', ab_error.item())
        plt.plot(res)
        plt.plot(y)
        # plt.show()

