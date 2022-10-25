import torch, numpy as np

from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Embedding
from torch.nn import Parameter
from attention import SelfAttention
from attention import SelfAttentionSparse
from attentionpooling import AttentionPooling1D
from fastattention import FastAttention

from utils import MEME

class TISFM(Module):
  def __init__(self, num_of_response, motif_path, window_size, 
               stride = 1,
               pool_kernel_size = 2,
               pool_stride = 2):
    super(TISFM, self).__init__()

    #keep y dimension
    self.num_of_response = num_of_response

    #Reading motifs from a MEME file
    self.meme_file = MEME()
    self.motif_path = motif_path
    kernels = self.meme_file.parse(motif_path, "none")
    self.sigmoid = Sigmoid()

    #Extracting kernel dimensions for first convolutional layer definition
    out_channels, in_channels, kernel_size = kernels.shape

    #Positional embedding dictionary 
    self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)

    #First layer convolution. Weights set to motifs and are fixed.
    self.conv_motif = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    self.conv_motif.weight = torch.nn.Parameter(kernels)
    self.conv_motif.weight.requires_grad = False
    self.out1_length = int((window_size - kernel_size) / stride) + 1

    self.conv_bias = Parameter(torch.empty(1, out_channels, 1, requires_grad = True))
    self.conv_scale  = Parameter(torch.empty(out_channels, requires_grad = True))
    with torch.no_grad():
        torch.nn.init.ones_(self.conv_scale)
        torch.nn.init.zeros_(self.conv_bias)

    self.attentionpooling = AttentionPooling1D(self.out1_length * 2, out_channels//2, mode = "diagonal")

    #Final regression layer.
    self.linreg = Linear(in_features=int(out_channels/2), out_features=num_of_response)

    self.l = -1

  def init_weights(self):
    for name, layer in self.named_children():
      if name == "conv_motif":
        pass
      else:
        if isinstance(layer, Linear):
          torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
        elif isinstance(layer, Conv1d):
          torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "relu")
        else:
          pass

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))

  def unfreeze(self):
    self.conv_motif.weight.requires_grad = True

  def forward(self, x):

    l = x.shape[2]

    if self.l != l:
        ii_f = torch.arange(l//2, device = x.get_device())
        ii_r = torch.flip(torch.arange(l - l//2, device = x.get_device()),dims=[0])

        self.ii = torch.cat([ii_f, ii_r])
        self.l = l

    pos = self.position_emb(self.ii).view(1,1,-1)

    #positional embedding
    x = x + pos

    x = self.conv_motif(x)

    #convolution calibration
    x = torch.einsum("bcl, c -> bcl", x, self.conv_scale) + self.conv_bias

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1])

    #regression
    y = self.linreg(x)

    return y

  def motif_ranks(self):
    with torch.no_grad():
      final_layer = self.linreg.weight.cpu().detach().numpy()
    ii = np.argsort(-final_layer, axis=1)
    return ii, np.array(self.meme_file.motif_names())
