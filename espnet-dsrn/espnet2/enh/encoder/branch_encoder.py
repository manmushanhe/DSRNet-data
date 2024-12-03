import torch
import torch.nn as nn

from collections import OrderedDict
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.branch import Branch_Block

class BranchEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 1024,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Branch_Block() for _ in range(num_blocks))
        self._output_dim = input_size

    @property
    def output_dim(self) -> int:
        return self._output_dim
    


    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        output = input
        #output, ilens = self.blocks(input)
        for block in self.blocks:
            output, ilens = block(output, ilens)
  
        return output, ilens
