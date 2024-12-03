import torch
from espnet2.enh.encoder.mag_encoder import MagEncoder


encoder = MagEncoder()

x = torch.randn(1, 64, 126, 4)

conv = torch.nn.Conv2d(64, 128, (1,1), (1,1))
output = conv(x)
print(output.size())