import torch
import torch.nn as nn
import numpy as np
from espnet2.enh.encoder.Only_Noisy_Training_main.DCUnet10_TSTM.Dual_Transformer import Dual_Transformer
from espnet2.enh.encoder.Only_Noisy_Training_main.loss import RegularizedLoss
from espnet2.enh.encoder.abs_encoder import AbsEncoder

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        # x torch.Size([1, 512, 101, 2])
        x_real = x[..., 0]
        # print("x_real",x_real.shape) torch.Size([1, 512, 101])
        x_im = x[..., 1]
        # print(x_real.shape) torch.Size([5, 1, 512, 198])
        # print(x_im.shape)   torch.Size([5, 1, 512, 198])
        # raise
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
            
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)  # torch.Size([1, 512, 101])
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

# 模型的back bone
class DCUnet10(AbsEncoder):
    """
    Deep Complex U-Net.
    """
    def __init__(
        self, 
        n_fft: int = 1022,
        hop_length: int = 256,
        is_istft: bool = True,
    ):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_istft = is_istft
        self.loss_fn = RegularizedLoss()
        
        # downsampling/encoding
        # Encoder包含一个2维卷积一个批正则化一个激活函数
        # self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=45)
        # self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=45, out_channels=90)
        # self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        # self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        # self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # # upsampling/decoding
        # # ConvTranspose2d上采样卷积
        # self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        # self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=90)
        # self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=90)
        # self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=45)
        # self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, output_padding=(1,1),
        #                          out_channels=1, last_layer=True)
        self.downsample0 = Encoder(filter_size=(2,2), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(2,2), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(2,2), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(2,2), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(2,2), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        # ConvTranspose2d上采样卷积
        self.upsample0 = Decoder(filter_size=(2,2), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(2,2), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=90)
        self.upsample2 = Decoder(filter_size=(2,2), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=90)
        self.upsample3 = Decoder(filter_size=(2,2), stride_size=(2,2), in_channels=180, output_padding=(0,1), out_channels=45)
        self.upsample4 = Decoder(filter_size=(2,2), stride_size=(2,2), in_channels=90, output_padding=(0,1),
                                 out_channels=1, last_layer=True)
    # downsampling/encoding 
    #def forward(self, x, n_fft, hop_length,is_istft=True):
    def forward(self, x):
        #print(x.shape)             # torch.Size([5, 512, 198, 2])   
        x = torch.unsqueeze(x, 1)   # torch.Size([5, 1, 512, 198, 2])
        d0 = self.downsample0(x)    # d0: torch.Size([5, 45, 255, 98, 2])
        #print("d0:",d0.shape)
        d1 = self.downsample1(d0)   # d1: torch.Size([5, 90, 127, 48, 2])
        #print("d1:",d1.shape)
        d2 = self.downsample2(d1)   # d2: torch.Size([5, 90, 63, 23, 2])
        #print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    # d3: torch.Size([5, 90, 31, 11, 2])
        #print("d3:",d3.shape)    
        d4 = self.downsample4(d3)   # d4: torch.Size([5, 90, 15, 9, 2])
        #print("d4:",d4.shape)
        
        u0 = self.upsample0(d4)    # upsampling/decoding 
        #print("u0:",u0.shape)             # u0: torch.Size([5, 90, 31, 11, 2])
        if u0.shape[3]> d3.shape[3] :   
            u0 = u0[:,:,:,:d3.shape[3],:]
        c0 = torch.cat((u0, d3), dim=1)   # c0: torch.Size([5, 180, 31, 11, 2])
        #print("c0:",c0.shape)

        u1 = self.upsample1(c0)
        #print("u1:",u1.shape)

        if u1.shape[3]> d2.shape[3] :   
            u1 = u1[:,:,:,:d2.shape[3],:]

        c1 = torch.cat((u1, d2), dim=1)   # c1: torch.Size([5, 180, 63, 23, 2])
        #print("c1:",c1.shape)
                                        #  (63-1)*2 + 1*2 + 0 + 1
                                        #  22*2 + 2 +1 
        u2 = self.upsample2(c1)           # u2: torch.Size([5, 90, 127, 47, 2])
        #print("u2:",u2.shape)

        if u2.shape[3]> d1.shape[3] :   
            u2 = u2[:,:,:,:d1.shape[3],:]

        c2 = torch.cat((u2, d1), dim=1)
        #print("c2:",c2.shape)           # c2: torch.Size([5, 180, 127, 48, 2])
        u3 = self.upsample3(c2)
        #print("u3:",u3.shape)           # u3: torch.Size([5, 45, 255, 98, 2])

        if u3.shape[3]> d0.shape[3] :   
            u3 = u3[:,:,:,:d0.shape[3],:]
        c3 = torch.cat((u3, d0), dim=1)
        #print("c3:",c3.shape)           # c3: torch.Size([5, 90, 255, 98, 2])
        u4 = self.upsample4(c3)
        if u4.shape[3]> x.shape[3] :   
            u4 = u4[:,:,:,:x.shape[3],:]
        #print("u4:",u4.shape)           # u4: torch.Size([5, 1, 512, 198, 2])
        #print(u4.shape)
        #print(x.shape)
        output = u4 * x    # u4 - the mask


        if self.is_istft:
          output = torch.squeeze(output, 1)   #  torch.Size([5, 512, 198, 2]) 
          #print(output.shape) # torch.Size([5, 512, 198, 2])
          output = torch.istft(output, n_fft = self.n_fft, hop_length = self.hop_length, normalized=True)  
          #print(output.shape) #torch.Size([5, 50432])
        return output
    
    def subsample2(self,wav):  
        # This function only works for k = 2 as of now.
        k = 2
        channels, dim= np.shape(wav) 

        dim1 = dim // k -128     # 128 is used to correct the size of the sampled data, you can change it
        wav1, wav2 = np.zeros([channels, dim1]), np.zeros([channels, dim1])   # [2, 1, 32640]
        #print("wav1:", wav1.shape)
        #print("wav2:", wav2.shape)

        wav_cpu = wav.cpu()
        for channel in range(channels):
            for i in range(dim1):
                i1 = i * k
                num = np.random.choice([0, 1])
                if num == 0:
                    wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1], wav_cpu[channel, i1+1]
                elif num == 1:
                    wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+1], wav_cpu[channel, i1]

        return torch.from_numpy(wav1).cuda(), torch.from_numpy(wav2).cuda()
    @property
    def output_dim(self) -> int:
        return self.n_fft//2 +1



class DCUnet10_rTSTM(nn.Module):
    """
    Deep Complex U-Net with real TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, n_fft, hop_length,is_istft=True):
        # encoder
        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        # real TSTM
        d4_1 = d4[:, :, :, :, 0]
        d4_2 = d4[:, :, :, :, 1]
        d4_1 = self.dual_transformer(d4_1)
        d4_2 = self.dual_transformer(d4_2)

        out = torch.rand(d4.shape)
        out[:, :, :, :, 0] = d4_1
        out[:, :, :, :, 1] = d4_2
        out= out.to('cuda')

        # decoder
        u0 = self.upsample0(out)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        # 逆stft
        if is_istft:
          output = torch.squeeze(output, 1) 
          output = torch.istft(output, n_fft=n_fft, hop_length=hop_length, normalized=True) 
        
        return output

class DCUnet10_cTSTM(nn.Module):
    """
    Deep Complex U-Net with complex TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer_real = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]
        self.dual_transformer_imag = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    # downsampling/encoding 
    def forward(self, x, n_fft, hop_length,is_istft=True):
        # encoder
        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        # complex TSTM
        d4_real = d4[:, :, :, :, 0]
        d4_imag = d4[:, :, :, :, 1]

        out_real = self.dual_transformer_real(d4_real)- self.dual_transformer_imag(d4_imag)
        out_imag = self.dual_transformer_imag(d4_real) + self.dual_transformer_real(d4_imag)   
        
        out = torch.rand(d4.shape)
        out[:, :, :, :, 0] = out_real
        out[:, :, :, :, 1] = out_imag
        out= out.to('cuda')
        #print("out:",out.shape)  

        # decoder
        u0 = self.upsample0(out)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        if is_istft:
          output = torch.squeeze(output, 1)
          output = torch.istft(output, n_fft=n_fft, hop_length=hop_length, normalized=True)
        
        return output