import torch


class Concate(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.liner = torch.nn.Linear(514,257)

        
    def forward(self, enhanced: torch.Tensor, noisy: torch.Tensor):
        
        concatenate = torch.cat((enhanced,noisy),dim = 2)
        #print(concatenate.shape)
        output = self.liner(concatenate)
        #print(concatenate.shape)

        return output 






