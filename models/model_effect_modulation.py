import torch
import torch.nn
from torch.nn import ConvTranspose1d
from torch.nn import Conv1d

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

from models.template_model import TemplateModel
from attentionpooling import GlobalAttentionPooling1D
from effectmodulation import EffectModulation1d

class TISFM(TemplateModel):
    '''
        Simple network to find convolutions and then diffusing back to find the weak grammar.
    '''
    def __init__(self, num_of_response,
                       motif_path,
                       window_size,
                       stride = 1, pad_motif=4):
        super(TISFM, self).__init__(num_of_response, motif_path, window_size, stride, pad_motif,
                                    both_direction = False)

        self.activation = torch.nn.LeakyReLU()#torch.nn.GELU()
        self.activation_out = torch.nn.LeakyReLU()

        self.effect_diffusion = EffectModulation1d(self.out_channels,
                                                   [5, 10, 20, 50, 150])

        self.effect = Conv1d(in_channels = self.out_channels,
                                       out_channels = num_of_response,
                                       kernel_size = 1)

        self.pool = GlobalAttentionPooling1D(feature_size = num_of_response)


    def forward(self, x):
        '''
            After motif layer the effect is diffused back with transposed convolutional layers
            then at each layer the effects are mixed and then pooled for the final results.
        '''
        x = self.motif_layer(x)

        x = self.activation(x)

        x = self.effect_diffusion(x)

        x = self.activation(x)

        x = self.effect(x)

        x = self.activation_out(x)

        x = self.pool(x)[0]

        return x

if __name__ == "__main__":
    model = TISFM(8, "../local/dataset/immgen_atac/cisBP_mouse.meme", 300)
    x = torch.rand(256, 4, 300)
    assert model(x).shape == torch.Size([256, 8])