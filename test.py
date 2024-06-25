import os.path
import torch.cuda
from torch.utils.data import DataLoader
from Dataset.dataset_classification import DatasetECG, read_data
import argparse
from ST_Encoder.classification_model import MICN
from torch import optim
from utils import LabelSmoothingCrossEntropyLoss,EarlyStopping,make_dir,mf1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torchsummary import summary
from ST_Encoder.classification_model import TimeSeriesClassifier, TCN_Classifier
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='STMixup')

#-------  dataset settings ------------
parser.add_argument('--task',type=str,default='classification',
                    choices=['classification','forecasting','anomaly'])
parser.add_argument('--direction',type=str,default='ECG5000')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--epoch',type=int,default=500)
parser.add_argument('--seqlen',type=int,default=140)
parser.add_argument('--outlen',type=int,default=5)
parser.add_argument('--dmodel',type=int,default=64)
parser.add_argument('--mixlayer',type=int,default=0)
parser.add_argument('--part',type=str,default='Trend')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = 'Data/UCRArchive_2018'
direction = args.direction
filename = args.direction+'_TRAIN.tsv'
filepath = path+'/'+direction+'/'+filename
test_data,test_label = read_data(filepath,val=False)
test_dataset = DatasetECG(test_data,test_label,device=device)
test_loader = DataLoader(test_dataset,batch_size=500,shuffle=False,drop_last=False)
model = MICN(dec_in=1,c_out=1,seq_len=args.seqlen,out_len=args.outlen,tcn_kernel=3,d_model=args.dmodel,dropout=0.2,mode='TCN',part=args.part,layer_mix=args.mixlayer).to(device)
pklpath = 'checkpoints/classification/epoch500_bs64_lr0.001_dm64_ml2_Season/best_model.pkl'
model.load_state_dict(torch.load(pklpath))
acc = 0

for x,y in test_loader:
    with torch.no_grad():
        model.eval()
        yhat = model(x)
        total = y.size(0)
        yhat = yhat.argmax(dim=1)
        y = y.argmax(dim=1)
        yhat = yhat.cpu().numpy()
        y = y.cpu().numpy()
        f1 = f1_score(y,yhat,average='macro')
        correct = (yhat == y).sum().item()
        acc += correct / total
print('f1 score:',f1)
print('accuracy:',acc)