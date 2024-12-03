import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class NawrDecoder(AbsDecoder):
    """Transposed Convolutional decoder for speech enhancement and separation"""

    def __init__(self, use_adaptive_weight: bool = True):
        super().__init__()

        self.use_adaptive_weight = use_adaptive_weight

        self.liner_enhanced0 = torch.nn.Linear(257,257,False)
        self.liner_noise0 = torch.nn.Linear(257,257,False)
        self.liner_add0_enh = torch.nn.Linear(257,257)
        self.liner_add0_noise = torch.nn.Linear(257,257)
        self.relu0 = torch.nn.ReLU()
        

        self.liner_enhanced1 = torch.nn.Linear(257,257,False)
        self.liner_noise1 = torch.nn.Linear(257,257,False)
        self.liner_add1_enh = torch.nn.Linear(257,257)
        self.liner_add1_noise = torch.nn.Linear(257,257)
        self.relu1 = torch.nn.ReLU()
        

        self.bn = torch.nn.BatchNorm1d(1,eps=1e-9,momentum=0.001,affine=True,track_running_stats=True)

    def forward(self, enhanced: torch.Tensor, noise: torch.Tensor):
        

        B , T , F = enhanced.shape

        enhanced_pre = torch.zeros_like(enhanced)
        noise_pre = torch.zeros_like(noise)

        ratio = torch.sum(enhanced, (1,2)) / torch.sum(noise, (1,2))
        
        ratio = ratio.reshape(B,1)
        ratio = self.bn(ratio)
        print(ratio)
        raise
        #f.write("ratio:"+str(ratio)+"\n")
        #f.write("ratio:"+str(self.bn.weight)+"\n")
        #f.write("ratio:"+str(self.bn.weight.device)+"\n")
        for i in range(B):
            if ratio[i] > mid :
                enhanced1 = self.liner_enhanced1(enhanced[i])
                enhanced1 = self.relu1(enhanced1)
                #enhanced1 = self.dropout1(enhanced1)
                
                noise1 = self.liner_noise1(noise[i])
                noise1 = self.relu1(noise1)
                #noise1 = self.dropout1(noise1)

                enh_residual1 = self.liner_add1_enh(noise1 + enhanced1)
                #enh_residual1 = self.dropout1(enh_residual1)

                noise_residual1 = self.liner_add1_noise(noise1 + enhanced1)
                #noise_residual1 = self.dropout1(noise_residual1)
                
                enhanced_pre[i] = enh_residual1 + enhanced[i]
                noise_pre[i] = noise_residual1 +  noise[i]

            else:
                enhanced0 = self.liner_enhanced0(enhanced[i])
                enhanced0 = self.relu0(enhanced0)
                #enhanced0 = self.dropout0(enhanced0)

                noise0 = self.liner_noise0(noise[i])
                noise0 = self.relu0(noise0)
                #noise0 = self.dropout0(noise0)

                enh_residual0 = self.liner_add0_enh(noise0 + enhanced0)
                #enh_residual0 = self.dropout0(enh_residual0)

                noise_residual0 = self.liner_add1_noise(noise0 + enhanced0)
                #noise_residual0 = self.dropout0(noise_residual0)
                
                enhanced_pre[i] = enh_residual0 + enhanced[i]
                noise_pre[i] = noise_residual0 +  noise[i]


        return enhanced_pre, noise_pre 


    def forward_loss(self, enhanced: torch.Tensor, noise_pre: torch.Tensor, clean: torch.Tensor , noisy: torch.Tensor):
        
        assert enhanced.shape == clean.shape == noisy.shape, (
            enhanced.shape,
            clean.shape,
            noisy.shape,
        )
        noise = noisy - clean

        error_speech = torch.abs(enhanced - clean)
        error_noise = torch.abs(noise_pre - noise)

        mse_speech = error_speech**2
        mse_noise = error_noise**2

        mse_speech = mse_speech.mean(dim=[1, 2])
        mse_noise = mse_noise.mean(dim=[1, 2])
        

        mse_speech = torch.mean(mse_speech)
        mse_noise = torch.mean(mse_noise)
        
        a = torch.mean(error_speech)/torch.mean(error_speech+error_noise)
        
        #f=open("/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/error_two_branch.txt",'a')
        #f.write("mse_speech:"+str(mse_speech)+"\n")
        #f.write("mse_noise:"+str(mse_noise)+"\n")
        #f.write("a:"+str(a)+"\n")
        #raise
        if self.use_adaptive_weight:
            mse = a*mse_speech + (1-a)*mse_noise
        else:
            mse = 0.5*mse_speech + 0.5*mse_noise

        return mse