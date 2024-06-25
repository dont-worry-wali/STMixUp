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
parser.add_argument('--direction',type=str,default='ETTh1')
parser.add_argument('--lr',type=float,default=1e-2)
parser.add_argument('--batchsize',type=int,default=1024)
parser.add_argument('--epoch',type=int,default=500)
parser.add_argument('--seqlen',type=int,default=96)
parser.add_argument('--outlen',type=int,default=96)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--dmodel',type=int,default=32)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='Season',
                    choices=['Raw','Trend','Season','Final','None','both','cmix'])
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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = args.gpu if torch.cuda.is_available() else 'cpu'
path = 'Data/all_six_datasets'
filepath = os.path.join(path,args.direction)
filepath = os.path.join(filepath,args.direction+'.csv')
train_data, test_data = read_data2(filepath, rate=args.rate, univariate=args.univariate,chosen_list=args.chosen_list)
if args.univariate:
    dim = 1
else:
    dim = train_data.shape[-1]

if args.scale == 'std':
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data-mean)/std
    test_data = (test_data-mean)/std
if args.scale == 'minmax':
    data_min = train_data.min(axis=0)
    data_max = train_data.max(axis=0)
    train_data = (train_data-data_min)/(data_max-data_min)
    test_data = (test_data-data_min)/(data_max-data_min)
length = len(train_data)


model_name = 'checkpoints/forecasting/ETTh1/ETTh1_epoch500_bs1024_lr0.01_dm32_None_decomp[32]_dmodel32/43/best_model0.pkl'

if args.model == 'micn':
    model = MICN(dec_in=dim, c_out=dim, seq_len=args.seqlen, out_len=args.outlen, tcn_kernel=args.tcn_kernel,
                 d_model=args.dmodel, dropout=args.dropout, mode='TCN', part=args.part,
                 layer_mix=args.mixlayer).to(device)
else:
    model = tcnforecast(1, [16, 32, 64]).to(device)
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state_dict'])
saved_epoch = checkpoint['epoch']
test_dataset = DatasetETT(test_data, input_len=args.seqlen, output_len=args.outlen, step=96, device=device,
                          dim=dim)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
for x, y in test_loader:
    with torch.no_grad():
        model.eval()
        trend,season = model(x)
        trend, season, y = trend.squeeze(), season.squeeze(), y.squeeze()
        plt.plot(trend.cpu().numpy(), label='trend', c='r')
        plt.plot(season.cpu().numpy(), label='season', c='b')
        plt.plot(y.cpu().numpy(), label='y', c='g')
        plt.legend()

        plt.show()
        plt.close(1)