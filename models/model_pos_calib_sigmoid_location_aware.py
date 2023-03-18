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

    self.attentionpooling_module1 = AttentionPooling1D(self.conv_length , self.out_channels, mode = "diagonal")

    self.attentionpooling_module2 = AttentionPooling1D(int(np.ceil(self.conv_length/2)) , self.out_channels, mode = "diagonal")

    #self.attentionpooling_module3 = AttentionPooling1D(self.conv_length // 3 , self.out_channels, mode = "diagonal")

    interaction_layer_in = self.out_channels//2 + self.out_channels + self.out_channels * 2

    self.interaction_layer = Linear(in_features=interaction_layer_in, out_features=self.out_channels//2, bias=True)

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
    x = x + pos

    x = self.motif_layer(x)

    x = self.sigmoid(x)

    interactions_in = torch.cat([self.attentionpooling_module1(x)[0],
                                 self.attentionpooling_module2(x)[0]],dim=2)

    interactions_in = interactions_in.view(interactions_in.shape[0], -1)

    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention pooling
    x, _ = self.attentionpooling(x)

    x = x.view(x.shape[0], x.shape[1])

    interactions_in = torch.cat([x, interactions_in], dim=1)

    interactions = self.sigmoid(self.interaction_layer(interactions_in))
    x = x * interactions

    #regression
    y = self.linreg(x)

    return y
