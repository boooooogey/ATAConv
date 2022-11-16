import torch
import einops
import numpy as np

from torch.nn import Linear
from torch.nn import Conv1d
from torch.nn import ModuleList
from torch.nn import GELU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module

from attentionpooling import AttentionPooling1D

class FocalModulation1d(Module):
    def __init__(self, dim, focal_levels, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_levels = np.sort(np.unique(focal_levels))
        self.level_num = len(focal_levels)

        self.toquery = Linear(in_features=dim, out_features=dim, bias=bias)
        self.tovalue = Linear(in_features=dim, out_features=dim, bias=bias)

        self.togates = Linear(in_features=dim, out_features=self.level_num+1, bias=bias)

        self.outprojection = Linear(in_features=dim, out_features=dim, bias=True)

        self.activation = GELU()
        self.final_activation = Sigmoid()
        self.mask_activation = Softmax(dim=-1)
        self.focal = ModuleList()

        for kl in focal_levels:
            self.focal.append(Conv1d(in_channels=dim, out_channels=dim , kernel_size=kl, stride=1, groups=dim, padding=kl//2, bias=False))

        self.mix_depth = Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        b, c, l = x.shape

        focus = einops.einsum(x, self.tovalue.weight, "batch channel length, embedding channel -> batch embedding length") + self.tovalue.bias.view(1, -1, 1)
        gates = einops.einsum(x, self.togates.weight, "batch channel length,     gates channel -> batch     gates length") + self.togates.bias.view(1, -1, 1)
        x = einops.einsum(x, self.toquery.weight, "batch channel length, embedding channel -> batch embedding length") + self.toquery.bias.view(1, -1, 1) 

        focus = self.activation(self.focal[0](focus))
        focus_sum = einops.einsum(focus, gates[:, 0, :].view(b,l), "batch embedding length, batch length -> batch embedding length")
        for i, layer in enumerate(self.focal[1:], start=1):
            focus = self.activation(layer(focus))
            focus_sum = focus_sum + einops.einsum(focus, gates[:, i, :].view(b,l), "batch embedding length, batch length -> batch embedding length")
        global_focus = self.activation(torch.mean(focus, axis=2, keepdim=True))
        focus_sum = focus_sum + einops.einsum(global_focus, gates[:, self.level_num, :].view(b,l), "batch embedding length, batch length -> batch embedding length")
        focus_sum = self.mix_depth(focus_sum)
        mask = self.mask_activation(focus_sum)
        x = self.final_activation(x) * mask

        return x, mask#einops.einsum(x, self.outprojection.weight, "batch embedding length, channel embedding -> batch channel length") + self.outprojection.bias.view(1, -1, 1)

class FocalModulationviaPooling1d(Module):
    def __init__(self, dim, focal_levels, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_levels = np.sort(np.unique(focal_levels))
        self.level_num = len(focal_levels)

        self.togates = Linear(in_features=dim, out_features=self.level_num+2, bias=bias)

        self.activation = GELU()
        self.mask_activation = Softmax(dim=-1)
        self.channel_mask = Softmax(dim=-2)
        self.focal = ModuleList()

        for kl in focal_levels:
            self.focal.append(AttentionPooling1D(kernel_size=kl, feature_size=dim, keep_shape=True))

        self.mix_depth = Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        b, c, l = x.shape

        gates = einops.einsum(torch.max(x, axis=-1)[0], self.togates.weight, "batch channel, gates channel -> batch gates") + self.togates.bias.view(1, -1)
        gates = self.mask_activation(gates)

        focus_sum = einops.einsum(x, gates[:, 0], "batch embedding length, batch -> batch embedding length")
        for i, layer in enumerate(self.focal):
            focus_sum = focus_sum + einops.einsum(layer(x)[0], gates[:, i+1], "batch embedding length, batch -> batch embedding length")
        global_focus = torch.mean(x, axis=2, keepdim=True)
        focus_sum = focus_sum + einops.einsum(global_focus, gates[:, self.level_num+1], "batch embedding length, batch -> batch embedding length")

        focus_sum = self.mix_depth(focus_sum)

        return self.channel_mask(focus_sum) #einops.einsum(x, self.outprojection.weight, "batch embedding length, channel embedding -> batch channel length") + self.outprojection.bias.view(1, -1, 1)

class FocalModulationMask1d(Module):
    def __init__(self, dim, focal_levels, kernel_length, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_levels = np.sort(np.unique(focal_levels))
        self.level_num = len(focal_levels)
        self.kernel_length = kernel_length

        #self.tovalue = Linear(in_features=dim, out_features=dim, bias=bias)
        self.togates = Linear(in_features=dim, out_features=self.level_num+2, bias=bias)
        self.outprojection = Linear(in_features=dim, out_features=dim, bias=True)

        self.activation = GELU()
        self.final_activation = Sigmoid()
        #self.mask_activation = Softmax(dim=-1)
        self.focal = ModuleList()

        for kl in focal_levels:
            self.focal.append(Conv1d(in_channels=dim, out_channels=dim , kernel_size=kl, stride=1, groups=dim, padding=kl//2, bias=False))

        self.mix_depth = Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        self.final_pooling = AttentionPooling1D(kernel_size=kernel_length, feature_size=dim)

    def forward(self, x):
        b, c, l = x.shape

        gates = einops.einsum(torch.max(x, axis=-1)[0], self.togates.weight, "batch channel, gates channel -> batch gates") + self.togates.bias.view(1, -1)
        #gates = self.mask_activation(gates)

        focus = x
        focus_sum = einops.einsum(x, gates[:, 0], "batch embedding length, batch -> batch embedding length")
        for i, layer in enumerate(self.focal, start=1):
            focus = self.activation(layer(focus))
            focus_sum = focus_sum + einops.einsum(focus, gates[:, i], "batch embedding length, batch -> batch embedding length")
        global_focus = self.activation(torch.mean(focus, axis=2, keepdim=True))
        focus_sum = focus_sum + einops.einsum(global_focus, gates[:, self.level_num+1], "batch embedding length, batch -> batch embedding length")

        focus_sum = self.final_pooling(self.mix_depth(focus_sum))[0].view(b,c)

        return self.final_activation(self.outprojection(focus_sum))

