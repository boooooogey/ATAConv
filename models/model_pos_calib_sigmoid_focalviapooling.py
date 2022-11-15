import torch, numpy as np

from torch.nn import Sigmoid
from attentionpooling import AttentionPooling1D
from focalmodulation import FocalModulationviaPooling1d

from models.template_model import TemplateModel

class TISFM(TemplateModel):
  def __init__(self, num_of_response, motif_path, window_size, 
               stride = 1, pad_motif=4):
    super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

    self.sigmoid = Sigmoid()

    self.focal_layer = FocalModulationviaPooling1d(self.out_channels//2, [5, 10, 25, 50, 75, 100])

    self.attentionpooling = AttentionPooling1D(self.conv_length * 2, self.out_channels//2, mode = "diagonal")

    self.l = -1

  def forward(self, x):

    x = self.motif_layer(x)

    x = self.sigmoid(x)
    x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

    #attention on convolution output
    x = x + self.focal_layer(x)[0]

    #attention pooling
    x, _ = self.attentionpooling(x)
    x = x.view(x.shape[0], x.shape[1])

    #regression
    y = self.linreg(x)

    return y
