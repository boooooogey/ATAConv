import torch

from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import BatchNorm1d
from torch.nn import ReLU
from torch.nn import ParameterList
from torch import flatten
from transformer import TransformerBlock
from transformer import SelfAttention

from utils import MEME

class MaskNet(Module):
  def __init__(self, num_of_response, motif_path, window_size, num_heads, stride = 1,
               conv_num_channels = 256, # For the convolutional layers except for the first one
               conv_kernel_size = 16, # For the convolutional layers except for the first one
               pool_kernel_size = 2,
               pool_stride = 2):
    super(MaskNet, self).__init__()

    #keep y dimension
    self.num_of_response = num_of_response

    #Attention heads
    self.num_heads = num_heads

    #Reading motifs from a MEME file
    self.meme_file = MEME()
    self.motif_path = motif_path
    kernels = self.meme_file.parse(motif_path, "none")

    #Extracting kernel dimensions for first convolutional layer definition
    out_channels, in_channels, kernel_size = kernels.shape

    self.out1_length = int((window_size - kernel_size) / stride) + 1

    reduced_dim = 256

    #Transformer layer
    self.attention = SelfAttention(out_channels, reduced_dim, self.num_heads)

    #First layer convolution. Weights set to motifs and are fixed.
    self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    self.conv1.weight = torch.nn.Parameter(kernels)
    self.conv1.weight.requires_grad = False

    #Rest of the model
    self.relu1 = ReLU()

    #Second convolutional layer
    self.conv2 = Conv1d(in_channels=reduced_dim, 
                        out_channels=conv_num_channels, 
                        kernel_size=conv_kernel_size, stride=stride)

    self.out2_length = int((self.out1_length - conv_kernel_size) / stride) + 1

    self.relu2 = ReLU()

    self.maxpool1 = MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    self.out3_length = int((self.out2_length - pool_kernel_size)/pool_stride) + 1

    #Linear Layer
    self.lin1 = Linear(in_features=self.out3_length * conv_num_channels, out_features=self.out1_length)
    self.relu3 = ReLU()

    self.maxpoolmask = MaxPool1d(kernel_size=self.out1_length * 2, stride=1)

    #Final regression layer. Only layer that does not share weights
    self.linreg = ParameterList([Linear(in_features=int(out_channels/2), out_features=1, bias=False) for i in range(num_of_response)])

  def init_weights(self):
    for name, layer in self.named_children():
      if name == "conv1":
        pass
      else:
        if isinstance(layer, Linear):
          torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
        elif isinstance(layer, Conv1d):
          torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "relu")
        elif isinstance(layer, ParameterList):
          for li in layer:
            torch.nn.init.kaiming_uniform_(li.weight, nonlinearity = "linear")
        else:
          pass

  def weight_reset(self):
    for name, layer in self.name_children():
      if name == "conv1":
        pass
      else:
        if isinstance(layer, Linear) or isinstance(layer, Conv1d):
          layer.reset_parameters()
        elif isinstance(layer, ParameterList):
          for li in layer:
            li.reset_parameters()

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))

  def forward(self, x):

    x = self.conv1(x)
    mask = self.relu1(x)

    mask = self.attention(mask)

    #Mask computations
    #mask = self.batchnorm1(x)

    mask = self.conv2(mask)
    #mask = self.batchnorm2(mask)
    mask = self.relu2(mask)
    mask = self.maxpool1(mask)

    mask = flatten(mask, 1)
    mask = self.lin1(mask)
    mask = self.relu3(mask)

    #Applying mask on the result of the first convolution
    dim1, _, dim2 = x.shape
    mask = mask.view(dim1, 1, dim2)
    y = mask * x

    #maxpooling is run over 2 consecutive channels, 5' and 3' directions.
    y = self.maxpoolmask(y.view(y.shape[0], int(y.shape[1]/2), int(y.shape[2]*2)))
    y = y.view(y.shape[0], y.shape[1])
    
    #regression
    y = torch.hstack([l(y).view(-1,1) for l in self.linreg])

    return y
    
  def masked(self, x):

    x = self.transformer(x)

    x = self.conv1(x)

    #Mask computations
    #mask = self.batchnorm1(x)
    mask = self.relu1(x)

    mask = self.conv2(mask)
    #mask = self.batchnorm2(mask)
    mask = self.relu2(mask)
    mask = self.maxpool1(mask)

    mask = flatten(mask, 1)
    mask = self.lin1(mask)
    mask = self.relu3(mask)

    #Applying mask on the result of the first convolution
    dim1, _, dim2 = x.shape
    mask = mask.view(dim1, 1, dim2)
    y = mask * x

    return y
    
class MaskNetprev(Module):
  def __init__(self, num_of_response, motif_path, window_size, num_heads, stride = 1,
               conv_num_channels = 256, # For the convolutional layers except for the first one
               conv_kernel_size = 16, # For the convolutional layers except for the first one
               pool_kernel_size = 2,
               pool_stride = 2):
    super(MaskNetprev, self).__init__()

    #keep y dimension
    self.num_of_response = num_of_response

    #Attention heads
    self.num_heads = num_heads

    #Reading motifs from a MEME file
    self.meme_file = MEME()
    self.motif_path = motif_path
    kernels = self.meme_file.parse(motif_path, "none")

    #Extracting kernel dimensions for first convolutional layer definition
    out_channels, in_channels, kernel_size = kernels.shape

    self.out1_length = int((window_size - kernel_size) / stride) + 1

    #Transformer layer
    self.transformer = TransformerBlock(in_channels, self.num_heads)

    #First layer convolution. Weights set to motifs and are fixed.
    self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    self.conv1.weight = torch.nn.Parameter(kernels)
    self.conv1.weight.requires_grad = False

    #Rest of the model
    #self.batchnorm1 = BatchNorm1d(out_channels)
    self.relu1 = ReLU()
    #self.maxpool1 = MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    #self.out2_length = int((self.out1_length - pool_kernel_size)/pool_stride) + 1

    #Second convolutional layer
    self.conv2 = Conv1d(in_channels=out_channels, 
                        out_channels=conv_num_channels, 
                        kernel_size=conv_kernel_size, stride=stride)

    #self.out3_length = int((self.out2_length - conv_kernel_size) / stride) + 1
    self.out2_length = int((self.out1_length - conv_kernel_size) / stride) + 1

    #self.batchnorm2 = BatchNorm1d(conv_num_channels)
    self.relu2 = ReLU()
    #self.maxpool2 = MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    #self.out4_length = int((self.out3_length - pool_kernel_size)/pool_stride) + 1

    #Third convolutional layer
    #self.conv3 = Conv1d(in_channels=conv_num_channels, 
    #                    out_channels=conv_num_channels, 
    #                    kernel_size=conv_kernel_size, stride=stride)

    #self.out5_length = int((self.out4_length - conv_kernel_size) / stride) + 1

    #self.relu3 = ReLU()
    #self.maxpool3 = MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    self.maxpool1 = MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    #self.out6_length = int((self.out5_length - pool_kernel_size)/pool_stride) + 1
    self.out3_length = int((self.out2_length - pool_kernel_size)/pool_stride) + 1

    #Linear Layer
    #self.lin1 = Linear(in_features=self.out6_length * conv_num_channels, out_features=self.out1_length)
    self.lin1 = Linear(in_features=self.out3_length * conv_num_channels, out_features=self.out1_length)
    self.relu3 = ReLU()

    # self.out1_length * 2 to max pool for both direction ( motif reverse and forward )
    self.maxpoolmask = MaxPool1d(kernel_size=self.out1_length * 2, stride=1)

    #Final regression layer. Only layer that does not share weights
    self.linreg = ParameterList([Linear(in_features=int(out_channels/2), out_features=1, bias=False) for i in range(num_of_response)])

  def init_weights(self):
    for name, layer in self.named_children():
      if name == "conv1":
        pass
      else:
        if isinstance(layer, Linear):
          torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
        elif isinstance(layer, Conv1d):
          torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "relu")
        elif isinstance(layer, ParameterList):
          for li in layer:
            torch.nn.init.kaiming_uniform_(li.weight, nonlinearity = "linear")
        else:
          pass

  def weight_reset(self):
    for name, layer in self.name_children():
      if name == "conv1":
        pass
      else:
        if isinstance(layer, Linear) or isinstance(layer, Conv1d):
          layer.reset_parameters()
        elif isinstance(layer, ParameterList):
          for li in layer:
            li.reset_parameters()

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))

  def forward(self, x):

    x = self.transformer(x)

    x = self.conv1(x)

    #Mask computations
    #mask = self.batchnorm1(x)
    mask = self.relu1(x)

    mask = self.conv2(mask)
    #mask = self.batchnorm2(mask)
    mask = self.relu2(mask)
    mask = self.maxpool1(mask)

    mask = flatten(mask, 1)
    mask = self.lin1(mask)
    mask = self.relu3(mask)

    #Applying mask on the result of the first convolution
    dim1, _, dim2 = x.shape
    mask = mask.view(dim1, 1, dim2)
    y = mask * x

    #maxpooling is run over 2 consecutive channels, 5' and 3' directions.
    y = self.maxpoolmask(y.view(y.shape[0], int(y.shape[1]/2), int(y.shape[2]*2)))
    y = y.view(y.shape[0], y.shape[1])
    
    #regression
    y = torch.hstack([l(y).view(-1,1) for l in self.linreg])

    return y
    
  def masked(self, x):

    x = self.transformer(x)

    x = self.conv1(x)

    #Mask computations
    #mask = self.batchnorm1(x)
    mask = self.relu1(x)

    mask = self.conv2(mask)
    #mask = self.batchnorm2(mask)
    mask = self.relu2(mask)
    mask = self.maxpool1(mask)

    mask = flatten(mask, 1)
    mask = self.lin1(mask)
    mask = self.relu3(mask)

    #Applying mask on the result of the first convolution
    dim1, _, dim2 = x.shape
    mask = mask.view(dim1, 1, dim2)
    y = mask * x

    return y
    
#  def forward(self, x):
#
#    x = self.transformer(x)
#
#    x = self.conv1(x)
#
#    #Mask computations
#    mask = self.relu1(x)
#    mask = self.maxpool1(mask)
#
#    mask = self.conv2(mask)
#    mask = self.relu2(mask)
#    mask = self.maxpool2(mask)
#
#    mask = self.conv3(mask)
#    mask = self.relu3(mask)
#    mask = self.maxpool3(mask)
#
#    mask = flatten(mask, 1)
#    mask = self.lin1(mask)
#    mask = self.relu4(mask)
#
#    #Applying mask on the result of the first convolution
#    dim1, _, dim2 = x.shape
#    mask = mask.reshape(dim1, 1, dim2)
