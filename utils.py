import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted



class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()


    def forward(self, input, target):
        loss = F.kl_div(input.log_softmax(dim=-1), target, reduction='batchmean')
        return loss


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
        print("Checkpoint saved")


def make_dir(args,seed):

    filename = args.direction+'_epoch'+str(args.epoch)+'_'+'bs'+str(args.batchsize)+'_'+'lr'+str(args.lr)+'_dm'+str(args.dmodel)+'_'\
               +args.part+'_decomp'+str(args.decomp)+'_dmodel'+str(args.dmodel)
    filepath = os.path.join('results', args.task)
    filepath = os.path.join(filepath, args.direction)
    if os.path.exists(filepath):
        pass
    else:
        os.mkdir(filepath)
    filepath = os.path.join(filepath, filename)
    modelpath = os.path.join('checkpoints', args.task)
    modelpath = os.path.join(modelpath, args.direction)
    if os.path.exists(modelpath):
        pass
    else:
        os.mkdir(modelpath)
    modelpath = os.path.join(modelpath,filename)

    if os.path.exists(filepath):
        pass
    else:
        os.mkdir(filepath)
    filepath = os.path.join(filepath,str(seed))
    if os.path.exists(filepath):
        print(filepath+' has been occupied')
        pass
    else:
        os.mkdir(filepath)

    if os.path.exists(modelpath):
        pass
    else:
        os.mkdir(modelpath)
    modelpath = os.path.join(modelpath,str(seed))
    if os.path.exists(modelpath):
        print(modelpath+' has been occupied')
        pass
    else:
        os.mkdir(modelpath)
    return filepath,modelpath


def mf1_score(y_true,y_pred,average='macro'):
    y_true = y_true.argmax(dim=1).cpu().numpy()
    y_pred = y_pred.argmax(dim=1).cpu().numpy()

    f1 = f1_score(y_true,y_pred,average=average)

    return f1


def accuracy(predictions, targets):
    """
    计算准确率
    Args:
        predictions (torch.Tensor): 模型的预测值，形状为 (batch_size, sequence_length)
        targets (torch.Tensor): 真实的标签，形状为 (batch_size, sequence_length)

    Returns:
        float: 准确率
    """
    if predictions.shape != targets.shape:
        raise ValueError("预测值和真实标签的形状不一致")

    targets = targets.argmax(dim=1).cpu().numpy()
    predictions = predictions.argmax(dim=1).cpu().numpy()


    correct = (predictions == targets).sum().item()
    total = len(predictions)

    accuracy = correct / total
    return accuracy


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted, target):
        loss = torch.sqrt(self.mse(predicted, target))
        return loss

def mse_loss(predicted, target):
    criterion = nn.MSELoss()
    loss = criterion(predicted, target)
    return loss

def mae_loss(predicted, target):
    criterion = nn.L1Loss()
    loss = criterion(predicted, target)
    return loss


def compute_saliency_maps(data, model):
    """
    Compute saliency maps for a batch of data using the given model.

    Parameters:
    - data: Tensor of shape (batch_size, channels, sequence_length), the input data.
    - model: A PyTorch model that accepts `data` as input and outputs predictions.

    Returns:
    - saliency_maps: Tensor of shape (batch_size, sequence_length), saliency scores for each time step in each sample.
    """
    # Ensure gradients can be computed
    data.requires_grad_()

    # Forward pass
    outputs = model(data)

    # Assume we're interested in the highest score for computing saliency
    score_max_index = outputs.argmax(dim=1)
    score_max = outputs.gather(1, score_max_index.unsqueeze(1)).squeeze()

    # Backward pass
    model.zero_grad()
    score_max.sum().backward()  # Sum needed to produce scalar for backward

    # Saliency map is the absolute value of the gradient
    saliency_maps = data.grad.abs().sum(dim=2)  # Sum over channels

    return saliency_maps




def saliency_mix_improved(data_a, target, model, lam):
    """
    Apply an improved Saliency Mix to a batch of time series data.

    Parameters:
    - data_a, data_b: Tensors of shape (batch_size, channels, sequence_length).
    - model: A trained PyTorch model for computing saliency maps.
    - alpha: Parameter for the beta distribution to control the mixing ratio.
    - lam: The proportion of the segment to mix from data_a to data_b.

    Returns:
    - mixed_data: The result of applying the improved Saliency Mix to the batch.
    """
    batch_size, sequence_length, channels = data_a.shape
    index = torch.randperm(batch_size).to('cuda:1')

    # Compute saliency maps for data_a
    saliency_maps_a = compute_saliency_maps(data_a, model)

    mixed_data = data_a[index, :]
    for i in range(batch_size):
        # Find the index of the maximum saliency point
        max_saliency_index = saliency_maps_a[i].argmax().item()


        # Calculate the start and end indices of the segment to mix
        segment_length = int(sequence_length * lam)
        start_index = max(0, max_saliency_index - segment_length // 2)
        end_index = start_index + segment_length

        # Adjust if the segment goes beyond the sequence length
        if end_index > sequence_length:
            end_index = sequence_length
            start_index = end_index - segment_length

        # Mix the segment from data_a to data_b
        mixed_data[i, start_index:end_index, :] = data_a[i, start_index:end_index, :]
    mixed_y = lam * target + (1 - lam) * target[index, :]

    return mixed_data,mixed_y

def saliencymix(data_a, target, model, lam):
    batch_size, sequence_length, channels = data_a.shape
    index = torch.randperm(batch_size).to('cuda:1')

    # Compute saliency maps for data_a
    saliency_maps_a = compute_saliency_maps(data_a, model)
    mixed_data = data_a[index, :]

    selectnum = int(sequence_length * lam)

    for i in range(batch_size):
        # Find the index of the maximum saliency point
        sorted_saliency, indices = torch.sort(saliency_maps_a[i], descending=True)
        selectindices = indices[:selectnum]


        # Adjust if the segment goes beyond the sequence length v
        # Mix the segment from data_a to data_b
        mixed_data[i, selectindices, :] = data_a[i, selectindices, :]
    mixed_y = lam * target + (1 - lam) * target[index, :]

    return mixed_data, mixed_y

def generate_low_freq_mask(sequence_length, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_freq = int(sequence_length * cut_rat) // 2  # 保留一半为低频部分

    # 生成低频掩码
    mask = torch.zeros(sequence_length,device='cpu')
    mask[:cut_freq] = 1  # 低频部分设置为1
    # 由于FFT结果是对称的，需要对后半部分的对称位置也设置为1
    mask[-cut_freq:] = 1
    return mask

def fmix_time_series(data, target,lam, device):
    batch_size, sequence_length = data.shape[0], data.shape[1]
    mask = generate_low_freq_mask(sequence_length, lam)
    data = data.cpu().numpy()
    origin = data[0]
    plt.plot(origin,c='orange',label='org1')

    # 对数据执行FFT
    data_fft = np.fft.fft(data,axis=1)

    # 随机选择另一组数据进行混合
    indices = torch.randperm(batch_size).to(device)
    data_fft_mixed = data_fft.copy()
    mask = mask.unsqueeze(0).unsqueeze(-1)  # 从[sequence_length]变形为[1, 1, sequence_length]
    mask = mask.numpy()
    origin2 = data[indices[0]]
    plt.plot(origin2,c='b',label='org2')

    # 应用掩码进行混合
    for i in range(batch_size):
        data_fft_mixed[i] = lam * data_fft[i] + (1 - lam) * data_fft[indices[i]] * mask

    # 执行逆FFT，将数据转换回时间域
    mixed_data = np.fft.ifft(data_fft_mixed).real  # 取实部，因为原始数据是实数
    mixdata = mixed_data[0]
    plt.plot(mixdata,c='g',label='mixed')
    plt.legend()
    mixed_data = torch.tensor(mixed_data,device=device,dtype=torch.float32)
    mixed_y = lam * target + (1 - lam) * target[indices, :]

    return mixed_data, mixed_y

def vanilla_mixup(data,target,lam):
    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to('cuda:1')
    mixed_out = lam * data + (1 - lam) * data[index, :]
    mixed_y = lam * target + (1 - lam) * target[index, :]
    return mixed_out, mixed_y


def cutmix(X1,y1,lam):
    """
    CutMix implementation for time series data.

    Parameters:
    X1 : numpy.ndarray
        Input time series data of shape (num_samples, num_timesteps, num_features) for first batch.
    X2 : numpy.ndarray
        Input time series data of shape (num_samples, num_timesteps, num_features) for second batch.
    y1 : numpy.ndarray
        Labels for the first batch.
    y2 : numpy.ndarray
        Labels for the second batch.
    beta : float, optional
        Hyperparameter for controlling the extent of cutmix. Default is 1.0.

    Returns:
    X_cutmixed : numpy.ndarray
        CutMixed time series data.
    y_cutmixed : numpy.ndarray
        Labels for the CutMixed data.
    """
    num_samples, num_timesteps, num_features = X1.shape

    rand_index = np.random.permutation(num_samples)
    index = torch.randperm(num_samples).to('cuda:1')

    cut_length = int(lam * num_timesteps)

    X_cutmixed = X1[index]
    y_cutmixed = y1[index]

    for i in range(num_samples):
        # Randomly choose the center of the cut segment
        center = np.random.randint(num_timesteps)

        # Calculate the start and end of the cut segment
        cut_start = max(0, center - cut_length // 2)
        cut_end = min(num_timesteps, center + cut_length // 2)

        # Perform CutMix
        X_cutmixed[i, cut_start:cut_end, :] = X1[i, cut_start:cut_end, :]
        y_cutmixed[i] = lam * y1[i] + (1 - lam) * y_cutmixed[i]

    return X_cutmixed, y_cutmixed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='STMixup')
    # -------  dataset settings ------------
    parser.add_argument('--task', type=str, default='forecasting',
                        choices=['classification', 'forecasting', 'anomaly'])
    parser.add_argument('--direction', type=str, default='ETTh1')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seqlen', type=int, default=96)
    parser.add_argument('--outlen', type=int, default=96)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--dmodel', type=int, default=64)
    parser.add_argument('--mixlayer', type=int, default=0)
    parser.add_argument('--part', type=str, default='None',
                        choices=['Raw', 'Trend', 'Season', 'Final', 'None'])
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--tcn_kernel', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rate', type=float, default=0.8)
    parser.add_argument('--univariate', type=bool, default=True)
    parser.add_argument('--chosen_list', type=int, default=-1)
    parser.add_argument('--checkpoint_num', type=int, default=5)
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--scale', type=str, default='std', choices=['std', 'minmax'])
    parser.add_argument('--test', type=bool, default=True)
    args = parser.parse_args()

    filepath,modelpath = make_dir(args,43)
    print(filepath)
    print(modelpath)

