import torch


class Dsrn_conv2d(torch.nn.Module):
    def __init__(self, use_adaptive_weight: bool = False, use_adaptive_weight_mae = False, use_asymmetric = False):
        super().__init__()

        self.use_adaptive_weight = use_adaptive_weight
        self.use_adaptive_weight_mae = use_adaptive_weight_mae
        self.use_asymmetric = use_asymmetric

        self.conv_enhanced1 = torch.nn.Conv2d(1,32,(1,13),1,(0,6))
        self.conv_noise1 = torch.nn.Conv2d(1,32,(1,13),1,(0,6))

        self.conv_enhanced2 = torch.nn.Conv2d(32,32,(1,13),1,(0,6))
        self.conv_noise2 = torch.nn.Conv2d(32,32,(1,13),1,(0,6))

        self.conv_enh_down = torch.nn.Conv2d(32,1,(1,13),1,(0,6))
        self.conv_noise_down = torch.nn.Conv2d(32,1,(1,13),1,(0,6))

        self.conv_enhanced1_2 = torch.nn.Conv2d(1,32,(1,13),1,(0,6))
        self.conv_noise1_2 = torch.nn.Conv2d(1,32,(1,13),1,(0,6))

        self.conv_enhanced2_2 = torch.nn.Conv2d(32,32,(1,13),1,(0,6))
        self.conv_noise2_2 = torch.nn.Conv2d(32,32,(1,13),1,(0,6))

        self.conv_enh_down2 = torch.nn.Conv2d(32,1,(1,13),1,(0,6))
        self.conv_noise_down2 = torch.nn.Conv2d(32,1,(1,13),1,(0,6))
        self.relu1 = torch.nn.ReLU()
    



    def forward(self, enhanced: torch.Tensor, noise: torch.Tensor):
        
        enhanced = enhanced.unsqueeze(1)    # (B,1,T,F)
        noise = noise.unsqueeze(1)  # (B,1,T,F)
        enhanced1 = self.conv_enhanced1(enhanced)   # (B,32,T,F)
        enhanced1 = self.relu1(enhanced1)   # (B,32,T,F)

        
        enhanced1 = self.conv_enhanced2(enhanced1)  # (B,32,T,F)
        enhanced1 = self.relu1(enhanced1)

        noise1 = self.conv_noise1(noise)
        noise1 = self.relu1(noise1)


        noise1 = self.conv_noise2(noise1)
        noise1 = self.relu1(noise1)


        enh_residual1 = self.conv_enh_down(noise1 + enhanced1) # (B,1,T,F)

        noise_residual1 = self.conv_noise_down(noise1 + enhanced1)  # (B,1,T,F)
                        
        enhanced_pre = enh_residual1 + enhanced
        noise_pre = noise_residual1 +  noise

        enh_residual1 = self.conv_enhanced1_2(enhanced_pre)
        enh_residual1 = self.relu1(enh_residual1)

        noise_residual1 = self.conv_noise1_2(noise_pre)
        noise_residual1 = self.relu1(noise_residual1)

        enh_residual1 = self.conv_enhanced2_2(enh_residual1)
        enh_residual1 = self.relu1(enh_residual1)

        noise_residual1 = self.conv_noise2_2(noise_residual1)
        noise_residual1 = self.relu1(noise_residual1)

        enh_residual1 = self.conv_enh_down(noise_residual1 + enh_residual1) # (B,1,T,F)

        noise_residual1 = self.conv_noise_down(noise_residual1 + enh_residual1)  # (B,1,T,F)

        enhanced_pre = enh_residual1 + enhanced_pre
        noise_pre = noise_residual1 +  noise_pre
        

        enhanced_pre = torch.squeeze(enhanced_pre,1)
        noise_pre = torch.squeeze(noise_pre,1)

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


    
