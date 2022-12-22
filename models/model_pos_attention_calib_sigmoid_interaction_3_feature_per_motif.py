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
    super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif, 6)

    self.sigmoid = Sigmoid()

    self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)

    self.attention_layer = SelfAttentionSparse(4, heads=2)

    self.attentionpooling = AttentionPooling1D(int(np.ceil(self.conv_length / 3)), self.out_channels, mode = "diagonal")

    self.interaction_layer = Linear(in_features=self.out_channels*3, out_features=self.out_channels*3, bias=True)

    self.l = -1

  def forward(self, x):

    l = x.shape[2]

    if self.l != l:
        ii_f = torch.arange(l//2, device = x.get_device())
        ii_r = torch.flip(torch.arange(l - l//2, device = x.get_device()), dims=[0])

        self.ii = torch.cat([ii_f, ii_r])
        self.l = l

    pos = self.position_emb(self.ii).view(1,1,-1)

    #positional embedding
    attention = self.attention_layer(x + pos)
    x = x + attention

    x = self.motif_layer(x)

    x = self.sigmoid(x)
    #x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1]*x.shape[2])

    interactions = self.sigmoid(self.interaction_layer(x))
    x = x * interactions

    #regression
    y = self.linreg(x)

    return y
