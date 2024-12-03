import torch


class Dsrn(torch.nn.Module):
    def __init__(self, use_adaptive_weight: bool = False, use_adaptive_weight_mae = False, use_asymmetric = False):
        super().__init__()

        self.use_adaptive_weight = use_adaptive_weight
        self.use_adaptive_weight_mae = use_adaptive_weight_mae
        self.use_asymmetric = use_asymmetric

        self.liner_enhanced1 = torch.nn.Linear(257,257,False)
        self.liner_noise1 = torch.nn.Linear(257,257,False)
        self.liner_add1_enh = torch.nn.Linear(257,257)
        self.liner_add1_noise = torch.nn.Linear(257,257)
        self.relu1 = torch.nn.ReLU()
    



    def forward(self, enhanced: torch.Tensor, noise: torch.Tensor):
        

        enhanced1 = self.liner_enhanced1(enhanced)
        enhanced1 = self.relu1(enhanced1)

                
        noise1 = self.liner_noise1(noise)
        noise1 = self.relu1(noise1)


        enh_residual1 = self.liner_add1_enh(noise1 + enhanced1)


        noise_residual1 = self.liner_add1_noise(noise1 + enhanced1)

                
        enhanced_pre = enh_residual1 + enhanced
        noise_pre = noise_residual1 +  noise

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

        
        # size[]
        mse_speech = torch.mean(error_speech**2)
        mse_noise = torch.mean(error_noise**2)
        
        a = torch.mean(error_speech)/torch.mean(error_speech+error_noise)
        
        if self.use_adaptive_weight:
            loss = a*mse_speech + (1-a)*mse_noise

        elif self.use_adaptive_weight_mae:
            mae_speech = torch.mean(error_speech)
            mae_noise = torch.mean(error_noise)
            loss = a*mae_speech + (1-a)*mae_noise

        elif self.use_asymmetric:
            error_speech_low_band = self.relu1(clean[:,:,0:40] - enhanced[:,:,0:40])
            error_speech_high_band = torch.abs(clean[:,:,40:257] - enhanced[:,:,40:257])
            error_speech = torch.cat((error_speech_low_band,error_speech_high_band),dim = 2)
            
            asymmetric_mse_speech = torch.mean(error_speech**2)

            a = torch.mean(error_speech)/torch.mean(error_speech+error_noise)
            loss = a*asymmetric_mse_speech + (1-a)*mse_noise
        else :
            loss = 0.5*mse_speech + 0.5*mse_noise

        return loss


    

