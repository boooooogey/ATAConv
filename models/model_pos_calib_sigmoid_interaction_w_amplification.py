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

    self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)

    self.attentionpooling = AttentionPooling1D(self.conv_length * 2, self.out_channels//2, mode = "diagonal")

    self.interaction_layer_mute = Linear(in_features=self.out_channels//2, out_features=self.out_channels//2, bias=True)
    self.interaction_layer_amplify = Linear(in_features=self.out_channels//2, out_features=self.out_channels//2, bias=True)

    self.l = -1

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

    x = self.motif_layer(x)

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1])

    mute = self.sigmoid(self.interaction_layer_mute(x))
    amplify= self.sigmoid(self.interaction_layer_amplify(x))

    x = x * mute + (1 - x) * amplify 

    #regression
    y = self.linreg(x)

    return y
