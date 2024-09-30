"""
Implementation of Nystrom convolution. Nystrom approximation is applied in a slide window manner
"""
import torch
from torch import Tensor
from torch import no_grad 
from torch.nn import Conv1d
from torch.nn import Module

from einops import einsum

def inverse_sqrt(mat: Tensor.float) -> Tensor.float:
    """Calculates the square root matrix of the inverse of a matrix. 

    Args:
        mat (Tensor.Float): input matrix, A

    Returns:
        Tensor.Float: A^(-1/2)
    """
    eig = torch.linalg.eig(mat)
    inv_sqrt = einsum(eig.eigenvectors, 1/torch.sqrt(eig.eigenvalues), eig.eigenvectors,
                      "i k, k, j k -> i j")
    return torch.real(inv_sqrt)

class NysConv1d(Module):
    """
    1 dimensional convolutional layer works on Nystrom approximation principle.
    """
    def __init__(self, representation_dim: int,
                 kernel_size: int)-> None:
                 # TODO figure out how to apply kernel functions other than linear kernel. Then we
                 # can have arbitrary kernel and add the following line:
                 # kernel: Callable[[Tensor.float, Tensor.float], Tensor.float])-> None:
        """Constructor for Nystrom convolutional layer.

        Args:
            representation_dim (int): dimension of the representation space. 
            kernel_size (int): size of the sampling weights 
            kernel (Callable[[torch.float, torch.float], torch.float]): kernel function
        """
        super().__init__()

        self.representation_dim = representation_dim
        self.kernel_size = kernel_size

        self.conv_1d = Conv1d(in_channels = 1,
                              out_channels = representation_dim,
                              kernel_size = kernel_size)

    def forward(self, x: Tensor.float) -> Tensor.float:
        """Forward

        Args:
            x (torch.float): input of size (batch, channels, length)

        Returns:
            torch.float: returns (batch, channels, representation_dim, out_length)
        """
        batch, channels, length = x.shape

        x = self.conv_1d(x.view(batch * channels, 1, length)).view(batch,
                                                                   channels,
                                                                   self.representation_dim,
                                                                   -1)

        w = einsum(self.conv_1d.weight,
                   self.conv_1d.weight, "n i j, m i j -> n m")

        w = inverse_sqrt(w)

        return einsum(x, w, "b c r l, k r -> b c k l")

    def gram_approximation(self, x: Tensor.float) -> Tensor.float:
        """Approximate kernel matrix using learned archetypes

        Args:
            x (Tensor.float): input tensor of shape (batch, channel, length)

        Returns:
            Tensor.float: approximated kernel values along length. output shape:
            (batch, channel, channel, lenght) 
        """
        with no_grad():
            batch, channels, length = x.shape

            x = self.conv_1d(x.view(batch * channels, 1, length)).view(batch,
                                                                    channels,
                                                                    self.representation_dim,
                                                                    -1)

            w = einsum(self.conv_1d.weight,
                    self.conv_1d.weight, "n i j, m i j -> n m")

            w = inverse_sqrt(w)

            return einsum(x, w, x, "b c1 r l, k r, b c2 k l  -> b c1 c2 l")
