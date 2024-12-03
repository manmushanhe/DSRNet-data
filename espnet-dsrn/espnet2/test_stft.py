import torch
import torch.nn as nn
from layers.stft import Stft
from torch_complex.tensor import ComplexTensor
from espnet2.layers.log_mel import LogMel

x = torch.randn(1,16000)

stft = Stft(
    n_fft=512,
    win_length=512,
    hop_length=128,
    center=True,
    window="hann",
    normalized=False,
    onesided=True,
    )
input_stft, feats_lens = stft(x)
print('stft',input_stft.size())
input_real=input_stft[..., 0]
input_imaginary=input_stft[..., 1]

print('real',input_real.size())
print('imaginary',input_imaginary.size())

input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])

print('complex_stft',input_stft.size())
# spectrum
input_stft = input_stft.real**2 + input_stft.imag**2

input_magnitude=torch.sqrt(input_stft)

print('magnitude',input_magnitude.size())

input_power = input_magnitude**2

print('spectrogram',input_power.size())

# magnitude torch.Size([1, 126, 257])
conv1 = torch.nn.Conv2d(1, 64, (1,5), (1,2))
# conv1d torch.Size([64, 126, 127])
x = torch.randn(1,4,161)
output=conv1(x)

print('conv2d',output.size()) 
