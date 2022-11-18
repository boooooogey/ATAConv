import torch, numpy as np

from torch.nn import Sigmoid
from attentionpooling import AttentionPooling1D
from focalmodulation import FocalModulationMask1d
from torch.nn import Embedding

from models.template_model import TemplateModel

class TISFM(TemplateModel):
  def __init__(self, num_of_response, motif_path, window_size, 
               stride = 1, pad_motif=4):
    super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

    self.sigmoid = Sigmoid()

    self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)
    self.focal_layer = FocalModulationMask1d(self.out_channels//2, [3 + 4 * i for i in range(3)], self.conv_length*2)

    self.attentionpooling = AttentionPooling1D(self.conv_length * 2, self.out_channels//2, mode = "diagonal")

    self.l = -1

  def forward(self, x):

    l = x.shape[2]

    if self.l != l:
        if x.get_device() == -1:
            ii_f = torch.arange(l//2, device = torch.device("cpu"))
            ii_r = torch.flip(torch.arange(l - l//2, device = torch.device("cpu")), dims=[0])
        else:
            ii_f = torch.arange(l//2, device = x.get_device())
            ii_r = torch.flip(torch.arange(l - l//2, device = x.get_device()),dims=[0])
        self.ii = torch.cat([ii_f, ii_r])
        self.l = l

    pos = self.position_emb(self.ii).view(1,1,-1)

    x = self.motif_layer(x+pos)

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)
    mask = self.focal_layer(x)

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1])
    x = x + x * mask

    #regression
    y = self.linreg(x)

    return y
