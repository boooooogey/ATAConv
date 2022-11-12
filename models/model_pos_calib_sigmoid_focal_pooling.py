import torch, numpy as np

from torch.nn import Sigmoid
from focalpooling import FocalPooling1d

from models.template_model import TemplateModel

class TISFM(TemplateModel):
  def __init__(self, num_of_response, motif_path, window_size, 
               stride = 1, pad_motif=4):
    super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

    self.sigmoid = Sigmoid()

    self.focal_layer = FocalPooling1d(self.out_channels//2, [15 + 2 * i for i in range(6)])

  def forward(self, x):

    x = self.motif_layer(x)

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #Focal pooling
    x = self.focal_layer(x)

    #regression
    y = self.linreg(x)

    return y
