import os.path
import torch.cuda
from torch.utils.data import DataLoader
from Dataset.dataset_classification import data_ECG,DatasetECG
import argparse
from ST_Encoder.classification_model import STClass
from torch import optim
from utils import LabelSmoothingCrossEntropyLoss,make_dir,mf1_score,accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='STMixup')
#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='classification',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='ECG')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--epoch',type=int,default=50)
parser.add_argument('--dmodel',type=int,default=16)
parser.add_argument('--mixlayer',type=int,default=1)
parser.add_argument('--part',type=str,default='None',
                    choices=['Raw','Trend','Season','Final','None'])
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--tcn_kernel',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--rate',type=float,default=0.8)
parser.add_argument('--univariate',type=bool,default=True)
parser.add_argument('--chosen_list',type=int,default=-1)
parser.add_argument('--checkpoint_num',type=int,default=10)
parser.add_argument('--seed', type=int, nargs='+', default=[42,43,44,45,46])
parser.add_argument('--gpu',type=str,default='cuda:0')
parser.add_argument('--scale',type=str,default='std',choices=['std','minmax'])
parser.add_argument('--test',type=bool,default=True)
parser.add_argument('--num_class',type=int,default=5)
parser.add_argument('--valbatch',type=int,default=64)
parser.add_argument('--decomp',type=int,default=32)
args = parser.parse_args()


model_perform_total={}
for seed in args.seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.gpu if torch.cuda.is_available() else 'cpu'
    path = 'Data/ECGData'
    train_data,train_label,val_data,val_label,test_data,test_label = data_ECG(path,val=True,seed=seed,rate=args.rate)
    train_dataset = DatasetECG(train_data,train_label,device=device)
    val_dataset = DatasetECG(val_data,val_label,device=device)
    train_loader = DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=args.valbatch,shuffle=True,drop_last=False)
    model = STClass(1,args.num_class,[2,2,2],args.dmodel,args.decomp,0.5,args.part).to(device)
    criterion = LabelSmoothingCrossEntropyLoss()
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
        else:
            for x, y in train_loader:
                model.train()
                res,mixed_y = model(x,y)
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

                num += 1
        val_loss/=num
        val_mse/=num
        val_mae/=num
        val_losses.append(val_loss.item())
        writer.add_scalars('Loss/loss',{'Train':train_loss,'Val':val_loss},i)



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
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.savefig(os.path.join(filepath,'Loss.png'))

    if args.test:
        model_perform_acc={}
        model_perform_mf1={}
        for j in range(args.checkpoint_num):
            model_name = os.path.join(modelpath,'best_model{}.pkl'.format(j))
            model = STClass(1,args.num_class,[2,2,2],args.dmodel,args.decomp,0.5,args.part).to(device)
            checkpoint=torch.load(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            saved_epoch = checkpoint['epoch']
            test_dataset = DatasetECG(test_data,test_label,device=device)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)
            num = 0
            acc = 0
            mf1 = 0
            for x, y in test_loader:
                with torch.no_grad():
                    model.eval()
                    res = model(x)
                    loss_acc = accuracy(res, y)
                    loss_mf1 = mf1_score(y, res)
                    acc += loss_acc
                    mf1 += loss_mf1
                    num += 1
            acc /= num
            mf1 /= num
            model_perform_acc[saved_epoch]=acc
            model_perform_mf1[saved_epoch]=mf1
        max_key = max(model_perform_acc,key=model_perform_acc.get)
        max_acc = model_perform_acc[max_key]
        max_mf1 = model_perform_mf1[max_key]
        result = [max_key,max_acc,max_mf1]
        model_perform_total[seed] = result

if args.test:
    print('best performance on five experiments:',model_perform_total)

    accs=[]
    mf1s=[]
    for key,value in model_perform_total.items():
        accs.append(value[1])
        mf1s.append(value[2])
    accs = np.array(accs)
    mf1s = np.array(mf1s)
    mean_acc = accs.mean()
    mean_mf1 = mf1s.mean()
    std_acc = accs.std()
    std_mf1 = mf1s.std()
    print('test_acc:',mean_acc,'  ',std_acc)
    print('test_mf1:',mean_mf1,'  ',std_mf1)