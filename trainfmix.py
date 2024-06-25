import os.path
import torch.cuda
from torch.utils.data import DataLoader
from Dataset.dataset_forecasting import read_data2,DatasetETT
import argparse
from ST_Encoder.forecasting_model import MICN,tcnforecast,gruforecast,TimeSeriesTransformer
from torch import optim
from utils import RMSELoss,EarlyStopping,make_dir,mse_loss,mae_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from utils import vanilla_mixup,cutmix,saliencymix,saliency_mix_improved


parser = argparse.ArgumentParser(description='STMixup')
#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='forecasting',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='ETTh1')
parser.add_argument('--lr',type=float,default=4e-3)
parser.add_argument('--batchsize',type=int,default=256)
parser.add_argument('--epoch',type=int,default=500)
parser.add_argument('--seqlen',type=int,default=96)
parser.add_argument('--outlen',type=int,default=96)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--dmodel',type=int,default=16)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='saliency',
                    choices=['Raw','Trend','Season','Final','None','both','cutmix','saliency','sali'])
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--tcn_kernel',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--rate',type=float,default=0.8)
parser.add_argument('--univariate',type=bool,default=True)
parser.add_argument('--chosen_list',type=int,default=-1)
parser.add_argument('--checkpoint_num',type=int,default=5)
parser.add_argument('--seed', type=int, nargs='+', default=[42, 43, 44])
parser.add_argument('--gpu',type=str,default='cuda:1')
parser.add_argument('--scale',type=str,default='std',choices=['std', 'minmax'])
parser.add_argument('--test',type=bool,default=True)
parser.add_argument('--fftrate',type=float,default=0.01)
parser.add_argument('--mixfft',type=bool,default=False)
parser.add_argument('--decomp',type=int,nargs='+',default=[32])
parser.add_argument('--alpha',type=float,default=0.5)
parser.add_argument('--beta',type=float,default=0.5)
parser.add_argument('--valbatch',type=int,default=64)
parser.add_argument('--model', type=str, default='trans')
args = parser.parse_args()
print(vars(args))


model_perform_total={}
for seed in args.seed:
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
    train_data,val_data = train_data[:int(0.875*length)],train_data[int(0.875*length):]
    train_dataset = DatasetETT(train_data,input_len=args.seqlen,output_len=args.outlen,step=args.step,device=device,dim=dim)
    val_dataset = DatasetETT(val_data,input_len=args.seqlen,output_len=args.outlen,step=args.step,device=device,dim=dim)
    train_loader = DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=512,shuffle=True,drop_last=False)
    if args.model == 'micn':
        model = MICN(dec_in=dim, c_out=dim, seq_len=args.seqlen,out_len=args.outlen,tcn_kernel=args.tcn_kernel,d_model=args.dmodel,
                     dropout=args.dropout,mode='TCN',part=args.part,layer_mix=args.mixlayer,alpha=args.alpha,beta=args.beta,
                     gamma=args.fftrate,mixfft=args.mixfft).to(device)
    elif args.model == 'tcn':
        model = tcnforecast(1,[16,32,64,128]).to(device)
    elif args.model == 'gru':
        model = gruforecast(1,64,2).to(device)
    elif args.model == 'trans':
        model = TimeSeriesTransformer(1,96,256,4,3,2048).to(device)
    else:
        pass
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num = 0
    train_loss = 0
    val_loss = 0
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    save_interval = args.epoch//args.checkpoint_num


    filepath,modelpath = make_dir(args,seed)
    # early_stopping = EarlyStopping(patience=args.patience,delta=0,checkpoint_path=checkpoint_path)
    writer = SummaryWriter(filepath)

    for i in tqdm(range(args.epoch)):
        train_loss = 0
        val_loss = 0
        val_mae = 0
        val_mse = 0
        num = 0
        if args.part == 'None':
            for x,y in train_loader:
                model.train()
                res = model(x)
                loss = criterion(res,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num+=1
            train_loss/=num
            num = 0
            train_losses.append(train_loss.item())
        elif args.part == 'Raw':
            for x, y in train_loader:
                model.train()
                lam = np.random.beta(args.alpha, args.beta)
                mixed_data,mixed_y = vanilla_mixup(x,y,lam)
                res = model(mixed_data)
                loss = criterion(res,mixed_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num += 1
            train_loss /= num
            num = 0
            train_losses.append(train_loss.item())
        elif args.part == 'cutmix':
            for x, y in train_loader:
                model.train()
                lam = np.random.beta(args.alpha, args.beta)
                mixed_data,mixed_y = cutmix(x,y,lam)
                res = model(mixed_data)
                loss = criterion(res,mixed_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num += 1
            train_loss /= num
            num = 0
            train_losses.append(train_loss.item())
        elif args.part == 'saliency':
            for x, y in train_loader:
                model.train()
                lam = np.random.beta(args.alpha, args.beta)
                mixed_data,mixed_y = saliencymix(x,y,model,lam)
                res = model(mixed_data)
                loss = criterion(res,mixed_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num += 1
            train_loss /= num
            num = 0
            train_losses.append(train_loss.item())
        elif args.part == 'sali':
            for x, y in train_loader:
                model.train()
                lam = np.random.beta(args.alpha, args.beta)
                mixed_data,mixed_y = saliency_mix_improved(x,y,model,lam)
                res = model(mixed_data)
                loss = criterion(res,mixed_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num += 1
            train_loss /= num
            num = 0
            train_losses.append(train_loss.item())
        else:
            for x, y in train_loader:
                model.train()
                res, mixed_y = model(x, y)
                loss = criterion(res, mixed_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                num += 1
            train_loss /= num
            num = 0
            train_losses.append(train_loss.item())
        for x,y in val_loader:
            with torch.no_grad():
                model.eval()
                res = model(x)
                loss = criterion(res,y)
                val_loss += loss
                val_mae += mae_loss(res,y)
                val_mse += mse_loss(res,y)
                num += 1
        val_loss/=num
        val_mse/=num
        val_mae/=num
        val_losses.append(val_loss.item())
        writer.add_scalars('Loss/loss',{'Train':train_loss,'Val':val_loss},i)
        writer.add_scalars('Loss/Mae',{'Train':train_loss,'Mae':val_mae},i)
        writer.add_scalars('Loss/Mse',{'Train':train_loss,'Mse':val_mse},i)


        if i%save_interval==0:
            best_loss = float('inf')
            if val_loss.item()<best_loss:
                best_loss =val_loss.item()
                save_time = i//save_interval
                save_path = os.path.join(modelpath,'best_model{}.pkl'.format(save_time))
                torch.save({
                    'epoch':i,
                    'model_state_dict':model.state_dict()
                },save_path)
        else:
            if val_loss.item()<best_loss:
                best_loss =val_loss.item()
                save_time = i//save_interval
                save_path = os.path.join(modelpath,'best_model{}.pkl'.format(save_time))
                torch.save({
                    'epoch':i,
                    'model_state_dict':model.state_dict()
                },save_path)

        print('Current epoch:',i,'train_loss:',train_loss.item(),'val_loss:',val_loss.item())
        #     break
        # else:
        #     early_stopping.save_checkpoint(val_loss,model)


    writer.close()
    plt.figure(seed)
    plt.title('Loss')
    plt.plot(train_losses,c='r',label='train')
    plt.plot(val_losses,c='b',label='val')
    plt.ylim(0,1)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.legend()
    plt.savefig(os.path.join(filepath,'Loss.png'))

    if args.test:
        model_perform_mse={}
        model_perform_mae={}
        for j in range(args.checkpoint_num):
            model_name = os.path.join(modelpath,'best_model{}.pkl'.format(j))
            if args.model == 'micn':
                model = MICN(dec_in=dim,c_out=dim,seq_len=args.seqlen,out_len=args.outlen,tcn_kernel=args.tcn_kernel,
                             d_model=args.dmodel,dropout=args.dropout,mode='TCN',part=args.part,layer_mix=args.mixlayer).to(device)
            elif args.model == 'tcn':
                model = tcnforecast(1,[16,32,64,128]).to(device)
            elif args.model == 'gru':
                model = gruforecast(1,64,2).to(device)
            elif args.model == 'trans':
                model = TimeSeriesTransformer(1, 96, 256, 4, 3, 2048).to(device)
            else:
                pass
            checkpoint=torch.load(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            saved_epoch = checkpoint['epoch']
            test_dataset = DatasetETT(test_data, input_len=args.seqlen, output_len=args.outlen, step=1, device=device,dim=dim)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=False)
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
            model_perform_mse[saved_epoch]=mse
            model_perform_mae[saved_epoch]=mae
        min_key = min(model_perform_mse,key=model_perform_mse.get)
        min_mse = model_perform_mse[min_key]
        min_mae = model_perform_mae[min_key]
        result = [min_key,min_mse,min_mae]
        model_perform_total[seed] = result

if args.test:
    print('best performance on five experiments:',model_perform_total)

    mses=[]
    maes=[]
    for key,value in model_perform_total.items():
        mses.append(value[1])
        maes.append(value[2])
    mses = np.array(mses)
    maes = np.array(maes)
    mean_mse = mses.mean()
    mean_mae = maes.mean()
    std_mse = mses.std()
    std_mae = maes.std()
    print('test_mse:',mean_mse,'  ',std_mse)
    print('test_mae:',mean_mae,'  ',std_mae)
