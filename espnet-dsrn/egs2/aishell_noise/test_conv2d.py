import torch
repre = torch.randn(2,3,40)
fbank = torch.randn(2,3,40)
repre = torch.unsqueeze(repre, 2)
print(repre.shape)
fbank = torch.unsqueeze(fbank, 2)
print(fbank.shape)

cat = torch.cat((repre, fbank), dim=2)
print(cat.shape)

cat = torch.unsqueeze(cat, 1)
print(cat.shape)

#a = torch.randn(2,1,3,2,40)
conv_enhanced1 = torch.nn.Conv3d(1,1,kernel_size=(1,2,13),stride=1,padding=(0,0,6))
cat = conv_enhanced1(cat)
cat = torch.squeeze(cat, 1)
print(cat.shape)
cat = torch.squeeze(cat, 2)
print(cat.shape)