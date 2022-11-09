import torch
import einops
import numpy as np

from torch.nn import Linear
from torch.nn import Conv1d
from torch.nn import ModuleList
from torch.nn import GELU
from torch.nn import Module


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
        self.focal = ModuleList()

        for kl in focal_levels:
            self.focal.append(Conv1d(in_channels=dim, out_channels=dim , kernel_size=kl, stride=1, groups=dim, padding=kl//2, bias=False))

    def forward(self, x):
        b, c, l = x.shape

        query = einops.einsum(x, self.toquery.weight, "batch channel length, embedding channel -> batch embedding length") + self.toquery.bias.view(1, -1, 1) 
        focus = einops.einsum(x, self.tovalue.weight, "batch channel length, embedding channel -> batch embedding length") + self.tovalue.bias.view(1, -1, 1)
        gates = einops.einsum(x, self.togates.weight, "batch channel length,     gates channel -> batch     gates length") + self.togates.bias.view(1, -1, 1)

        focus = self.activation(self.focal[0](focus))
        focus_sum = einops.einsum(focus, gates[:, 0, :].view(b,l), "batch embedding length, batch length -> batch embedding length")
        for i, layer in enumerate(self.focal[1:], start=1):
            focus = self.activation(layer(focus))
            focus_sum = focus_sum + einops.einsum(focus, gates[:, i, :].view(b,l), "batch embedding length, batch length -> batch embedding length")
        global_focus = self.activation(torch.mean(focus, axis=2, keepdim=True))
        focus_sum = focus_sum + einops.einsum(global_focus, gates[:, self.level_num, :].view(b,l), "batch embedding length, batch length -> batch embedding length")

        return einops.einsum(x, self.outprojection.weight, "batch embedding length, channel embedding -> batch channel length") + self.outprojection.bias.view(1, -1, 1)

