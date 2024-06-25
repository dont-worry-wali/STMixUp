import os.path
import torch.cuda
from torch.utils.data import DataLoader
from Dataset.dataset_forecasting import read_data2,DatasetETT
import argparse
from ST_Encoder.forecasting_model import MICN
from torch import optim
from utils import RMSELoss,EarlyStopping,make_dir
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from utils import mae_loss,mse_loss


parser = argparse.ArgumentParser(description='STMixup')
#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='forecasting',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='ETTm1.csv')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--epoch',type=int,default=1)
parser.add_argument('--seqlen',type=int,default=96)
parser.add_argument('--outlen',type=int,default=96)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--dmodel',type=int,default=64)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='Trend')
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--tcn_kernel',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--rate',type=float,default=0.8)
parser.add_argument('--univariate',type=bool,default=True)
parser.add_argument('--chosen_list',type=int,default=-1)
parser.add_argument('--checkpoint_num',type=int,default=5)
parser.add_argument('--seed',type=int,nargs='+',default=[42,43,44,45,46])
parser.add_argument('--gpu',type=str,default='cuda')
parser.add_argument('--scale',type=str,default='std',choices=['std','minmax'])
args = parser.parse_args()

model_perform_total = {}
for seed in args.seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.gpu if torch.cuda.is_available() else 'cpu'
    path = 'Data'
    filepath = os.path.join(path,args.direction)
    train_data, test_data = read_data2(filepath,rate=args.rate,univariate=True,chosen_list=args.chosen_list)
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
    model_perform_mse = {}
    model_perform_mae = {}



    filepath = 'checkpoints/forecasting/ETTm1/ETTm1_epoch200_bs1024_lr0.002_dm64_ml0_Season_std'




    for j in range(args.checkpoint_num):
        # model_name = os.path.join(filepath, 'best_model{}.pkl'.format(j))
        model_name = filepath+'/'+str(seed)+'/'+'best_model{}.pkl'.format(j)
        model = MICN(dec_in=1, c_out=1, seq_len=args.seqlen, out_len=args.outlen, tcn_kernel=args.tcn_kernel,
                     d_model=args.dmodel, dropout=args.dropout, mode='TCN', part=args.part, layer_mix=args.mixlayer).to(
            device)
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint['epoch']
        test_dataset = DatasetETT(test_data, input_len=args.seqlen, output_len=args.outlen, step=1, device=device)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True)
        num = 0
        mae = 0
        mse = 0
        for x, y in test_loader:
            with torch.no_grad():
                model.eval()
                res = model(x)
                loss_mae = mae_loss(res, y)
                loss_mse = mse_loss(res, y)
                mae += loss_mae.item()
                mse += loss_mse.item()
                num += 1
        mae /= num
        mse /= num
        model_perform_mse[saved_epoch] = mse
        model_perform_mae[saved_epoch] = mae
    min_key = min(model_perform_mse, key=model_perform_mse.get)
    min_mse = model_perform_mse[min_key]
    min_mae = model_perform_mae[min_key]
    result = [min_key, min_mse, min_mae]
    model_perform_total[seed] = result

print('best performance on five experiments:', model_perform_total)

mses = []
maes = []
for key, value in model_perform_total.items():
    mses.append(value[1])
    maes.append(value[2])
mses = np.array(mses)
maes = np.array(maes)
mean_mse = mses.mean()
mean_mae = maes.mean()
std_mse = mses.std()
std_mae = maes.std()
print(filepath)
print('test_mse:', mean_mse, '  ', std_mse)
print('test_mae:', mean_mae, '  ', std_mae)
