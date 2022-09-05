import torch

from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch import flatten

from utils import MEME


class MaskNet(Module):
  def __init__(self, motif_path, window_size, stride = 1):

    super(MaskNet, self).__init__()

    #Reading motifs from a MEME file
    self.meme_file = MEME()
    kernels = self.meme_file.parse(motif_path, "none")

    #Extracting kernel dimensions for first convolutional layer definition
    out_channels, in_channels, kernel_size = kernels.shape

    self.out1_length = int((window_size - kernel_size) / stride) + 1

    #First layer. Weights set to motifs and are fixed.
    self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    self.conv1.weight = torch.nn.Parameter(kernels)
    self.conv1.weight.requires_grad = False

    #Rest of the model
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
    self.out2_length = int((self.out1_length - 2)/2) + 1

    #Second convolutional layer
    self.conv2 = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    self.out3_length = int((self.out2_length - kernel_size) / stride) + 1

    self.relu2 = ReLU()
    self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
    self.out4_length = int((self.out3_length - 2)/2) + 1

    #Linear Layer
    self.lin1 = Linear(in_features=self.out4_length * out_channels, out_features=self.out1_length)
    self.relu3 = ReLU()

    self.maxpoolmask = MaxPool1d(kernel_size=self.out1_length, stride=1)

    #Final regression layer
    self.linreg = Linear(in_features=out_channels, out_features=1)

  def forward(self, x):

    x = self.conv1(x)

    #Mask computations
    mask = self.relu1(x)
    mask = self.maxpool1(mask)

    mask = self.conv2(mask)
    mask = self.relu2(mask)
    mask = self.maxpool2(mask)

    mask = flatten(mask, 1)
    mask = self.lin1(mask)
    mask = self.relu3(mask)

    #Applying mask on the result of the first convolution
    dim1, _, dim2 = x.shape
    mask = mask.reshape(dim1, 1, dim2)
    y = mask * x
    y = self.maxpoolmask(y)
    y = y.reshape(y.shape[0], y.shape[1])
    
    #regression
    y = self.linreg(y)

    return y
    
