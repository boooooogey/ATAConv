import torch, numpy as np

from torch.nn import Sigmoid
from torch.nn import Embedding
from focalpooling import FocalPooling1d
from attention import SelfAttentionSparse

from models.template_model import TemplateModel

class TISFM(TemplateModel):
    def __init__(self, num_of_response, motif_path, window_size, 
                 stride = 1, pad_motif=4):
        super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

        self.sigmoid = Sigmoid()

        self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)
        self.attention_layer = SelfAttentionSparse(4, heads=2)

        self.focal_layer = FocalPooling1d(self.out_channels//2, [15 + 4 * i for i in range(3)])
        self.l = -1

    def forward(self, x):

        l = x.shape[2]

        if self.l != l:
            ii_f = torch.arange(l//2, device = x.get_device())
            ii_r = torch.flip(torch.arange(l - l//2, device = x.get_device()),dims=[0])

            self.ii = torch.cat([ii_f, ii_r])
            self.l = l

        pos = self.position_emb(self.ii).view(1,1,-1)
        x = x + self.attention_layer(x + pos)
          
        x = self.motif_layer(x)

        x = self.sigmoid(x)
        x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

        #Focal pooling
        x = self.focal_layer(x)

        #regression
        y = self.linreg(x)

        return y
