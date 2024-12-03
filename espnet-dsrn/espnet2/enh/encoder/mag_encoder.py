import torch
import torch.nn as nn

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.layers.stft import Stft
from collections import OrderedDict
from torch_complex.tensor import ComplexTensor


class MagEncoder(AbsEncoder):
    """Magnitude encoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_builtin_complex: bool = True,
    ):
        super().__init__()

        n_fft=512
        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
   
        self._create_encoder_blocks()
  
        #self.conv2 = torch.nn.Conv2d(
            #out_channels, out_channels, 1, 1, groups=pointwise_groups
        #)
        #self.conv3 = torch.nn.Conv2d(
            #out_channels, out_channels, 1, 1, groups=pointwise_groups
        #)

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
        
        input_stft, feats_lens = self._compute_stft(input)
        input_stft = input_stft.real**2 + input_stft.imag**2
        input_stft=torch.sqrt(input_stft)

        B, T, F = input_stft.size()

        input_frames = input_stft.view(B, 1, T, F)
        print('frames',input_frames.size())
        output = self.blocks.forward(input_frames)
       
        return output, feats_lens

    def gen_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list ,
        stride: list ,
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
        block["conv2d"] = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        block["batchnorm"] = torch.nn.BatchNorm2d(out_channels, affine=True)
        block["activation"] = torch.nn.ReLU()
        block["dropout"] = torch.nn.Dropout(dropout_probability)
        return torch.nn.Sequential(block)
    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
    def _create_encoder_blocks(self):
        blocks = OrderedDict()

        in_channels = [1, 64, 64, 64, 64]
        out_channels = [64, 64, 64, 64, 64]
        kernel_size = [[1,5], [1,3], [1,3], [1,3], [1,3]]
        stride = [1, 2]
        for layer in [1, 2, 3, 4, 5]:
            blocks[f"Encoder_block{layer}"] = self.gen_encoder_block(
                in_channels[layer-1], out_channels[layer-1], kernel_size=kernel_size[layer-1], stride=stride
        )

        self.blocks = torch.nn.Sequential(blocks)