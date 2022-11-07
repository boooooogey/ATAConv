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
from models.template_model import TemplateModel

class TISFM(TemplateModel):
  def __init__(self, num_of_response, motif_path, window_size, 
               stride = 1, pad_motif=4):
    super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

    self.sigmoid = Sigmoid()

    self.position_emb = Embedding(self.conv_length * 2, 1)

    self.attention_interaction = SelfAttentionSparse(self.out_channels//2, heads=4, inner_dim=50)

    self.attentionpooling = AttentionPooling1D(self.conv_length * 2, self.out_channels//2, mode = "diagonal")

    self.l = -1

  def forward(self, x):

    x = self.motif_layer(x)

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention on convolution output
    pos = self.position_emb(torch.arange(x.shape[2], device = x.get_device())).view(1,1,-1)
    x = x + self.attention_interaction(x + pos)

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1])

    #regression
    y = self.linreg(x)

    return y
