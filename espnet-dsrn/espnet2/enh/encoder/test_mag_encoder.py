import torch
from espnet2.enh.encoder.mag_encoder import MagEncoder


encoder = MagEncoder()

x = torch.randn(1,16000)

encoder = MagEncoder()
length = torch.Tensor([16000])
output, outputlengths = encoder.forward(x,length)
print(output.size)