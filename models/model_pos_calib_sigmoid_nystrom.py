import torch, numpy as np

from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Embedding
from attentionpooling import AttentionPooling1D
from nystrom_layer import Nystrom

from models.template_model import TemplateModel

class TISFM(TemplateModel):
    '''
        TISFM flavor that uses Nystrom features at the end instead of interactions.
    '''
    def __init__(self, num_of_response,
                       motif_path,
                       window_size,
                       stride = 1, pad_motif=4):
        super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif)

        self.sigmoid = Sigmoid()

        self.position_emb = Embedding(int(np.ceil(window_size/2)), 1)

        self.attentionpooling = AttentionPooling1D(self.conv_length * 2,
                                                   self.out_channels // 2,
                                                   mode = "diagonal")


        self.nystrom_layer = Nystrom(n_sample = 128)

        self.length = -1

        self.sym_indices = None

        self.linreg = Linear(in_features = 128, out_features = self.num_of_response)

    def forward(self, x):
        '''
            After convolutional and global pooling layers apply Nystrom method to find a feature map.
            The final regression layer is applied to the features obtain by Nystrom Method.
        '''

        length = x.shape[2]

        if self.length != length:
            device_ii = x.get_device()
            if device_ii < 0:
                device_ii = torch.device("cpu")
            ii_f = torch.arange(length // 2, device = device_ii)
            ii_r = torch.flip(torch.arange(length - length // 2, device = device_ii), dims = [0])

            self.sym_indices = torch.cat([ii_f, ii_r])
            self.length = length

        pos = self.position_emb(self.sym_indices).view(1,1,-1)

        #positional embedding
        x = x + pos

        x = self.motif_layer(x)

        x = self.sigmoid(x)
        x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

        #attention pooling
        x, _ = self.attentionpooling(x)
        x = x.view(x.shape[0], x.shape[1])

        x, _ = self.nystrom_layer(x)
        #regression
        x = self.linreg(x)

        return x

    def feature_map(self, x):
        '''
           Returns Nystrom features. 
        '''

        length = x.shape[2]

        device_ii = x.get_device()
        if device_ii < 0:
            device_ii = torch.device("cpu")
        ii_f = torch.arange(length // 2, device = device_ii)
        ii_r = torch.flip(torch.arange(length - length // 2, device = device_ii), dims = [0])

        self.sym_indices = torch.cat([ii_f, ii_r])
        self.length = length

        pos = self.position_emb(self.sym_indices).view(1,1,-1)

        #positional embedding
        x = x + pos

        x = self.motif_layer(x)

        x = self.sigmoid(x)
        x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

        #attention pooling
        x, _ = self.attentionpooling(x)
        x = x.view(x.shape[0], x.shape[1])

        x, cluster_centers = self.nystrom_layer(x)
        return x, cluster_centers
