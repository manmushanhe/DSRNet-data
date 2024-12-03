import torch
import torch.nn as nn
from espnet2.enh.encoder.Only_Noisy_Training_main.loss_utils import mse_loss, stftm_loss, reg_loss

time_loss = mse_loss()
freq_loss = stftm_loss()
reg_loss = reg_loss()

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False)
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0   # loss_mask.shape: torch.Size([2, 1, 32512])
    return loss_mask

class RegularizedLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()

        self.gamma = gamma

    '''
    def mseloss(self, image, target):
        x = ((image - target)**2)
        return torch.mean(x)
    '''

    def wsdr_fn(self, x_, y_pred_, y_true_, eps=1e-8):  # g1_wav, fg1_wav, g2_wav
        y_pred = y_pred_.flatten(1)
        y_true = y_true_.flatten(1)
        x = x_.flatten(1)

        def sdr_fn(true, pred, eps=1e-8):
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # true and estimated noise
        z_true = x - y_true
        z_pred = x - y_pred

        a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
        wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)

    def regloss(self, g1, g2, G1, G2):
        # print(g1.shape)
        # print(g2.shape)
        # print(G1.shape)
        # print(G1.shape)
        return torch.mean((g1-g2-G1+G2)**2)

    # g1_wav 
    def forward(self, g1_wav, fg1_wav, g2_wav, g1fx, g2fx):
        g1_wav = torch.unsqueeze(g1_wav, 1)  # #
        fg1_wav = torch.unsqueeze(fg1_wav, 1) #  torch.Size([5, 1, 50432]) 
        g2_wav = torch.unsqueeze(g2_wav, 1)   #  torch.Size([5, 1, 50624])
        g1fx = torch.unsqueeze(g1fx, 1)  # torch.Size([5, 1, 50560])
        g2fx = torch.unsqueeze(g2fx, 1)  # torch.Size([5, 1, 50560])
        B,C,L = g2_wav.shape
        L = g2_wav.shape[2] - g1fx.shape[2]
        #print(g2_wav.shape[2],g1fx.shape[1])
        #print(g1fx.shape)   # torch.Size([5, 1, 50560])
        #print(g2_wav.shape) # torch.Size([5, 1, 50624])
        pad = torch.zeros([B,1,L]).to('cuda')
        g1fx = torch.cat((g1fx, pad), dim=2)
        #print(g1fx.shape)
        g2fx = torch.cat((g2fx, pad), dim=2)
        #print(g2fx.shape)
        if(g2_wav.shape[0] == 2):
            nframes = [g2_wav.shape[2],g2_wav.shape[2]]   # nframes: [32512, 32512]
        else:
            nframes = [g2_wav.shape[2]]

        loss_mask = compLossMask(g2_wav, nframes)   
        loss_mask = loss_mask.float().cuda() 
        loss_time = time_loss(fg1_wav, g2_wav, loss_mask)
        loss_freq = freq_loss(fg1_wav, g2_wav, loss_mask)
        loss1 = (0.8 * loss_time + 0.2 * loss_freq)/600

        return loss1 + self.wsdr_fn(g1_wav, fg1_wav, g2_wav) + self.gamma * self.regloss(fg1_wav, g2_wav, g1fx, g2fx)
