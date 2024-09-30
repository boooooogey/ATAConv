import torch
from einops import einsum
import numpy as np
from torch import Tensor

from torch.nn import ModuleList
from torch.nn import LeakyReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn import ConvTranspose1d

from typing import List

class EffectModulation1d(Module):
    """
    Modulating the effect of a channel. The weights are applied in different levels. A gate is
    learned . 
    """
    def __init__(self,
                 dim: int,
                 focal_levels: List[int],
                 init_memory: float = 1,
                 bias: bool = True):
        """_summary_

        Args:
            dim (int): Number of channels 
            focal_levels (List[int]): Diffusion kernel length 
            bias (bool, optional): Should the layers have bias. Defaults to True.
        """
        super().__init__()
        self.dim = dim
        self.focal_levels = np.sort(np.unique(focal_levels))
        self.level_num = len(focal_levels)

        self.activation = LeakyReLU()
        self.mask_activation = Softmax(dim=-1)
        self.focal = ModuleList()
        self.memory = torch.nn.Parameter(torch.empty((dim,
                                                      len(focal_levels)), requires_grad = True))
        with torch.no_grad():
            self.memory[:] = init_memory

        for kl in focal_levels:
            self.focal.append(ConvTranspose1d(in_channels=dim,
                                              out_channels=dim ,
                                              kernel_size=kl,
                                              stride=1,
                                              groups=dim,
                                              bias=bias))

    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward operation for the layer.

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: effect diffusion
        """
        _, _, l = x.shape
        memory = self.mask_activation(self.memory)
        start_point = int(np.ceil(self.focal_levels[0]/2))
        effect_sum = einsum(self.activation(self.focal[0](x)[:,:,start_point:(start_point + l)]),
                            memory[:, 0],
                            "b c l, c -> b c l")
        for i, layer in enumerate(self.focal[1:], start = 1):
            start_point = int(np.ceil(self.focal_levels[i]/2))
            focus = einsum(self.activation(layer(x)[:,:,start_point:(start_point + l)]),
                                memory[:, i],
                                "b c l, c -> b c l")
            effect_sum += focus 
        return effect_sum

    def return_layers(self, x: Tensor.float) -> (Tensor.float, List[Tensor.float]):
        _, _, l = x.shape
        memory = self.mask_activation(self.memory)
        start_point = int(np.ceil(self.focal_levels[0]/2))
        out = list()
        effect = einsum(self.activation(self.focal[0](x)[:,:,start_point:(start_point + l)]),
                        memory[:, 0],
                        "b c l, c -> b c l")
        out.append(effect)
        effect_sum = effect
        for i, layer in enumerate(self.focal[1:], start = 1):
            start_point = int(np.ceil(self.focal_levels[i]/2))
            focus = einsum(self.activation(layer(x)[:,:,start_point:(start_point + l)]),
                                memory[:, i],
                                "b c l, c -> b c l")
            out.append(focus)
            effect_sum += focus
        return effect_sum, out