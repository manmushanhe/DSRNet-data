import torch
import torch.nn as nn

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from collections import OrderedDict

class LSTMEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 1024,
        num_layers: int = 2,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, (1,1), (1,1))
        self.norm1 = torch.nn.BatchNorm2d(128, affine=True)
        self.activation1 = torch.nn.ReLU()
        
        self.dconv1 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=1)
        self.dconv2 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=2)
        self.dconv3 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=4)
        self.dconv4 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=8)
        self.dconv5 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=16)
        self.dconv6 = torch.nn.Conv2d(64, 128, (1,1), (1,1),dilation=32)

        
        self.conv2 = torch.nn.Conv2d(128, 64, (1,1), (1,1))
        
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

        output, h = self.lstm(input)
        output = self.linear(output)
        output = self.activation(output)

    

        return output, ilens
    def gen_residual_block(
        self,
        dilation: int,
        dropout_probability: float = 0.2,
    ):
        """Generate a convolutional block for Lightweight Sinc convolutions.

        Each block consists of either a depthwise or a depthwise-separable
        convolutions together with dropout, (batch-)normalization layer, and
        an optional average-pooling layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the convolution.
            stride: Stride of the convolution.
            dropout_probability: Dropout probability in the block.

        Returns:
            torch.nn.Sequential: Neural network building block.
        """
        block = OrderedDict()
        block["1x1-conv"] = torch.nn.Conv2d(64, 128, (1,1), (1,1))
        block["activation"] = torch.nn.ReLU()
        block["batchnorm"] = torch.nn.BatchNorm2d(128, affine=True)
        block["D-conv"] = torch.nn.Conv2d(128, 128, (1,3), (1,1), dilation=dilation)
        block["activation"] = torch.nn.ReLU()
        block["batchnorm"] = torch.nn.BatchNorm2d(128, affine=True)
        block["1x1-conv"] = torch.nn.Conv2d(128, 64, (1,1), (1,1))
        return torch.nn.Sequential(block)
