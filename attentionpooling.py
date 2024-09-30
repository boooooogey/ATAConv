"""
Implementation of attention pooling
"""
import torch
from torch.nn import Module
from torch.nn import Parameter
from torch import Tensor
from einops import einsum, repeat

class AttentionPooling1D(Module):
    """Pools values using attention mechanism
    """
    def __init__(self,
                 kernel_size: int,
                 feature_size: int,
                 init_constant: int = 2,
                 keep_shape: bool = False,
                 mode: str = "diagonal" # "full", "diagonal", "shared"
                 ) -> None:
        """Constructor

        Args:
            kernel_size (int): kernel size for the pooling
            feature_size (int): number of channels that the expected input has.
            init_constant (int, optional): initial values for the pooling weights. Defaults to 2.
            keep_shape (bool, optional): should the dimensions of the input kept. Defaults to False.
            mode (str, optional): structure for the weight sampling. Defaults to "diagonal".
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.feature_size = feature_size
        self.mode = mode
        self.keep_shape = keep_shape

        if mode != "diagonal":
            self.out_feature = feature_size if mode != "shared" else 1 
            self.weight = Parameter(torch.empty(self.out_feature, feature_size, requires_grad = True)) 
            with torch.no_grad():
                if mode == "full":
                    torch.nn.init.eye_(self.weight)
                    self.weight.mul_(init_constant)
                if mode == "shared":
                    nn.init.xavier_uniform_(self.weight)
        else:
            self.out_feature = feature_size
            self.weight = Parameter(torch.empty(feature_size, requires_grad = True))
            with torch.no_grad():
                torch.nn.init.ones_(self.weight)
                self.weight.mul_(init_constant)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward function

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: output tensor 
        """
        b, c, l = x.shape
        original_length = l
        kernel_size = self.kernel_size
        if l % kernel_size != 0:
            reminder = kernel_size - l % kernel_size
            x = torch.nn.functional.pad(x, (0,reminder,0,0,0,0))
            _, _, l = x.shape

        x = x.view(b, c, l//kernel_size, kernel_size)

        if self.mode != "diagonal":
            mask = torch.einsum("bcld, ec -> beld", x, self.weight)
            mask = self.softmax(mask)
        else:
            mask = torch.einsum("bcld, c -> bcld", x, self.weight)
            mask = self.softmax(mask)

        if self.keep_shape:
            out = torch.sum(x * mask,
                            axis=-1).repeat_interleave(kernel_size, dim=-1)[:,:,:original_length]
        else:
            out = torch.sum(x * mask, axis=-1)

        return out, mask

class GlobalAttentionPooling1D(Module):
    """Reduces a dimension from the data by pooling values using attention mechanism
    """
    def __init__(self,
                 feature_size: int,
                 init_constant: int = 2,
                 mode: str = "diagonal" # "full", "diagonal", "shared"
                 ) -> None:
        """Constructor

        Args:
            kernel_size (int): kernel size for the pooling
            feature_size (int): number of channels that the expected input has.
            init_constant (int, optional): initial values for the pooling weights. Defaults to 2.
            keep_shape (bool, optional): should the dimensions of the input kept. Defaults to False.
            mode (str, optional): structure for the weight sampling. Defaults to "diagonal".
        """
        super().__init__()

        self.feature_size = feature_size
        self.mode = mode

        if mode != "diagonal":
            self.out_feature = feature_size if mode != "shared" else 1 
            self.weight = Parameter(torch.empty(self.out_feature, feature_size, requires_grad = True)) 
            with torch.no_grad():
                if mode == "full":
                    torch.nn.init.eye_(self.weight)
                    self.weight.mul_(init_constant)
                if mode == "shared":
                    nn.init.xavier_uniform_(self.weight)
        else:
            self.out_feature = feature_size
            self.weight = Parameter(torch.empty(feature_size, requires_grad = True))
            with torch.no_grad():
                torch.nn.init.ones_(self.weight)
                self.weight.mul_(init_constant)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward function

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: output tensor 
        """
        b, c, l = x.shape

        if self.mode != "diagonal":
            mask = torch.einsum("bcl, ec -> bel", x, self.weight)
            mask = self.softmax(mask)
        else:
            mask = torch.einsum("bcl, c -> bcl", x, self.weight)
            mask = self.softmax(mask)

        out = torch.sum(x * mask, axis=-1)

        return out, mask

class AttentionPooling2D(Module):
    """Pools values using attention mechanism from 2 dimensional data
    """
    def __init__(self,
                 kernel_size: int,
                 feature_size: int,
                 representation_size: int,
                 init_constant: int = 2) -> None:
        """Constructor

        Args:
            kernel_size (int): kernel size for the pooling
            feature_size (int): number of channels that the expected input has.
            representation_size (int): dimensions of the representation space for each feature.
            init_constant (int, optional): initial values for the pooling weights. Defaults to 2.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.feature_size = feature_size
        self.representation_size = representation_size

        self.out_feature = feature_size
        self.weight = Parameter(torch.empty((feature_size, representation_size), requires_grad = True))
        with torch.no_grad():
            torch.nn.init.ones_(self.weight)
            self.weight.mul_(init_constant)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward function

        Args:
            x (Tensor.float): input tensor

        Returns:
            Tensor.float: output tensor 
        """
        b, c, r, l = x.shape
        kernel_size = self.kernel_size
        if l % kernel_size != 0:
            reminder = kernel_size - l % kernel_size
            x = torch.nn.functional.pad(x, (0,reminder))
            _, _, _, l = x.shape

        x = x.view(b, c, r, l//kernel_size, kernel_size)

        mask = einsum(x, self.weight, "b c r l d, c r -> b c l d")
        mask = repeat(self.softmax(mask), "b c l d -> b c r l d", r = self.representation_size)

        out = torch.sum(x * mask, axis=-1)

        return out, mask
