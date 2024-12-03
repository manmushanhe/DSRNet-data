import torch

from espnet2.layers.stft import Stft

input = torch.rand(2,5120)
input_lengths = torch.Tensor([5120,5120])
print(input_lengths.shape) 

stft = Stft(
            n_fft=320,
            win_length=320,
            hop_length=128,
            center=True,
            window="hann",
            normalized=False,
            onesided=False,
            )

input_stft, feats_lens = stft(input, input_lengths)

print(input_stft.shape)
# center=True,onesided=True, torch.Size([2, 41, 161, 2])
# center=False,onesided=True, torch.Size([2, 38, 161, 2])
# center=False,onesided=False, torch.Size([2, 38, 320, 2])
# center=True,onesided=False,  torch.Size([2, 41, 320, 2])