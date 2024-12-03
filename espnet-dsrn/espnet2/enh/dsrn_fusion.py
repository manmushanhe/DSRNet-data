import torch



class Dsrn_fusion(torch.nn.Module):
    def __init__(self, use_adaptive_weight: bool = False):
        super().__init__()

        self.use_adaptive_weight = use_adaptive_weight

        self.liner_enhanced1 = torch.nn.Linear(257,257,False)
        self.liner_noise1 = torch.nn.Linear(257,257,False)
        self.liner_add1_enh = torch.nn.Linear(257,257)
        self.liner_add1_noise = torch.nn.Linear(257,257)
        self.relu1 = torch.nn.ReLU()
    

        #self.dropout1 = torch.nn.Dropout(p=0.2)



    def forward(self, enhanced: torch.Tensor, noise: torch.Tensor):
        

        enhanced1 = self.liner_enhanced1(enhanced)
        enhanced1 = self.relu1(enhanced1)

                
        noise1 = self.liner_noise1(noise)
        noise1 = self.relu1(noise1)


        enh_residual1 = self.liner_add1_enh(noise1 + enhanced1)


        noise_residual1 = self.liner_add1_noise(noise1 + enhanced1)

                
        enhanced_pre = enh_residual1 
        noise_pre = noise_residual1 

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
        
        a = torch.mean(torch.abs(error_speech))/torch.mean(torch.abs(error_speech)+torch.abs(error_noise))
        
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


    

