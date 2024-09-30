import torch
from einops import einsum
from torch import Tensor

from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Parameter

from attentionpooling import GlobalAttentionPooling1D

class AttentionMask1d(Module):
    """
    Calculates masks for a channel derived from the information contained in other channels.
    """
    def __init__(self,
                 num_of_channels: int,
                 bias: bool = True,
                 init: float = 5):
        """_summary_

        Args:
            num_of_channels (int): Number of channels 
            bias (bool, optional): Should the layers have bias. Defaults to True.
            init (float, optional): Starts the weights from this value.
        """
        super().__init__()
        self.num_of_channels = num_of_channels
        self.bias = bias

        self.weights = Linear(out_features = num_of_channels,
                              in_features = num_of_channels,
                              bias = bias)

        self.activation = Sigmoid()

        with torch.no_grad():
            self.weights.weight[:] = 0
            self.weights.weight.fill_diagonal_(init)
            if self.bias:
                self.weights.bias = Parameter(self.weights.bias.view(1, -1, 1))
                self.weights.bias[:] = 0



    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward operation for the layer.

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: effect diffusion
        """
        _, _, l = x.shape

        mask = einsum(self.weights.weight, x, "o i, b i l -> b o l")
        if self.bias:
            mask = mask + self.weights.bias
        mask = self.activation(mask)

        return x * mask

class AttentionMaskGlobal1d(Module):
    """
    Calculates masks for a channel derived from the information contained in other channels.
    This derives a single mask for the entire channel instead of computing a mask for each position
    separately.
    """
    def __init__(self,
                 num_of_channels: int,
                 bias: bool = True,
                 init: float = 5):
        """_summary_

        Args:
            num_of_channels (int): Number of channels 
            bias (bool, optional): Should the layers have bias. Defaults to True.
            init (float, optional): Starts the weights from this value.
        """
        super().__init__()
        self.num_of_channels = num_of_channels
        self.bias = bias

        self.weights = Linear(out_features = num_of_channels,
                              in_features = num_of_channels,
                              bias = bias)

        self.activation = Sigmoid()
        self.pooling = GlobalAttentionPooling1D(feature_size = num_of_channels)

        with torch.no_grad():
            self.weights.weight[:] = 0
            self.weights.weight.fill_diagonal_(init)
            if self.bias:
                self.weights.bias = Parameter(self.weights.bias.view(1, -1, 1))
                self.weights.bias[:] = 0



    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward operation for the layer.

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: effect diffusion
        """
        mask = einsum(self.weights.weight, x, "o i, b i l -> b o l")
        if self.bias:
            mask = mask + self.weights.bias

        mask = self.pooling(mask)[0]

        mask = self.activation(mask)

        return einsum(x, mask, "b c l, b c -> b c l")