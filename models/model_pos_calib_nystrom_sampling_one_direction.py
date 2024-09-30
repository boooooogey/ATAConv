import torch
from torch.nn import Sigmoid
from torch.nn import Linear

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

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
        super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif,
                                    both_direction = False)

        self.sigmoid = Sigmoid()

        feature_dim = 4

        self.nystrom_layer = Nystrom_Sampling(length = self.conv_length, feature_dim = feature_dim)

        self.linreg = Linear(in_features = self.out_channels * feature_dim, out_features = num_of_response)

    def forward(self, x):
        '''
            After convolutional and global pooling layers apply Nystrom method to find a feature
            map. The final regression layer is applied to the features obtain by Nystrom Method.
        '''
        x = self.motif_layer(x)

        x = self.sigmoid(x)

        x = self.nystrom_layer(x)

        x = x.view(x.shape[0], -1)

        #regression
        x = self.linreg(x)

        return x

    def feature_map(self, x):
        '''
            Return Nystrom feature map.
        '''
        x = self.motif_layer(x)

        x = self.sigmoid(x)

        return self.nystrom_layer(x), x

    def gram_mat(self, x):
        '''
            Return Nystrom approximated gram map.
        '''
        x = self.motif_layer(x)

        x = self.sigmoid(x)

        return self.nystrom_layer.gram_approximation(x)

if __name__ == "__main__":
    model = TISFM(8, "../local/dataset/immgen_atac/cisBP_mouse.meme", 300)
    x = torch.rand(256, 4, 300)
    print(model(x).shape)
