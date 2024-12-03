import torch
import torch.nn as nn

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.frequency_attention import Frequency_Attention


class Branch_Block(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 1024,
        kernel_size: int = 13,
        stride: int = 1,
        padding : int = 6,
    ):
        super().__init__()
        
        self.ln_att = torch.nn.LayerNorm(input_size)
        self.frequency_att = Frequency_Attention(257,0.5)
        
        self.ln_linear = torch.nn.LayerNorm(input_size)
        self.linear_projection0 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.relu = nn.ReLU()

        # 1024 = 1024 + 6 + 6 - 13 + 1
        self.ln_conv = torch.nn.LayerNorm(input_size)
        self.conv = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.linear_projection1 = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)
        self.dropout = nn.Dropout(p=0.5)
        
        self.liner_residual0 = torch.nn.Linear(257,257,False)
        self.liner_residual1 = torch.nn.Linear(257,257,False)
        self.liner_add = torch.nn.Linear(257,257)


        self._output_dim = input_size

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed feature [Batch, time, frequency]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        # branch0
        # ln_att
        #residual0 = self.ln_att(input)
        # att 
        residual0 = self.frequency_att(residual0)

        
        # branch1
        # ln_linear
        #residual1 = self.ln_linear(input)
        # projection0
        residual1 = self.linear_projection0(input)
        residual1 = self.relu(residual1)

        # ln_conv
        #residual1 = self.ln_conv(residual1)
        # conv
        residual1 = residual1.transpose(1, 2)
        residual1 = self.conv(residual1)
        residual1 = residual1.transpose(1, 2)
        # projection1
        residual1 = self.linear_projection1(residual1)
        # dropout
        residual1 = self.dropout(residual1)
        

        # merge
        # linear(branch0) 
        residual0 = self.liner_residual0(residual0)
        # linear(branch1) 
        residual1 = self.liner_residual1(residual1)
        # linear_add
        residual_merge = self.liner_add(residual0 + residual1)
        
        
        output = self.relu(input + residual_merge)
        

        return output, ilens
