import torch, numpy as np

from torch.nn import Sigmoid
from torch.nn import Linear
from torch.nn import Embedding
from einops import einsum
from attentionpooling import AttentionPooling1D
from nystrom_layer import Nystrom_Sampling

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

        self.nystrom_layer = Nystrom_Sampling(length = self.conv_length * 2, feature_dim = 2)

        self.linreg = Linear(in_features = self.out_channels, out_features = num_of_response)

        ii_f = torch.arange(window_size // 2)
        ii_r = torch.flip(torch.arange(window_size - window_size // 2), dims = [0])

        self.sym_indices = torch.cat([ii_f, ii_r])

    def forward(self, x):
        '''
            After convolutional and global pooling layers apply Nystrom method to find a feature map.
            The final regression layer is applied to the features obtain by Nystrom Method.
        '''
        input_dev = x.get_device()
        if self.sym_indices.get_device() != input_dev:
            if input_dev < 0:
              self.sym_indices = self.sym_indices.cpu()
            else:
              self.sym_indices = self.sym_indices.to(input_dev)
        pos = self.position_emb(self.sym_indices).view(1,1,-1)

        #positional embedding
        x = x + pos

        x = self.motif_layer(x)

        x = self.sigmoid(x)
        x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

        x = self.nystrom_layer(x)
        
        x = x.view(x.shape[0], -1)

        #regression
        x = self.linreg(x)

        return x
    
    def feature_map(self, x):
        '''
            Return Nystrom feature map.
        '''
        with torch.no_grad():
            input_dev = x.get_device()
            if self.sym_indices.get_device() != input_dev:
                if input_dev < 0:
                    self.sym_indices = self.sym_indices.cpu()
                else:
                    self.sym_indices = self.sym_indices.to(input_dev)
            pos = self.position_emb(self.sym_indices).view(1,1,-1)

            #positional embedding
            x = x + pos

            x = self.motif_layer(x)

            x = self.sigmoid(x)
            x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

            return self.nystrom_layer(x), x

    def gram_mat(self, x):
        '''
            Return Nystrom approximated gram map.
        '''
        with torch.no_grad():
            input_dev = x.get_device()
            if self.sym_indices.get_device() != input_dev:
                if input_dev < 0:
                    self.sym_indices = self.sym_indices.cpu()
                else:
                    self.sym_indices = self.sym_indices.to(input_dev)
            pos = self.position_emb(self.sym_indices).view(1,1,-1)

            #positional embedding
            x = x + pos

            x = self.motif_layer(x)

            x = self.sigmoid(x)
            x = x.view(x.shape[0], x.shape[1]//2, x.shape[2]*2)

            return self.nystrom_layer.gram_approximation(x)

if __name__ == "__main__":
  import torch
  model = TISFM(8, "../local/dataset/immgen_atac/cisBP_mouse.meme", 300)
  x = torch.rand(256, 4, 300)
  model(x)