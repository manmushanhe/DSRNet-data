import torch
import torch.nn as nn

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class LSTMEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 1024,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.4)
        #,bidirectional=True
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.activation = nn.ReLU()
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
