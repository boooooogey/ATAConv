import torch, numpy as np, math

from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import Parameter

from utils import MEME, return_coef_for_normalization
from scipy.optimize import fmin

from torch.nn.functional import pad

def mse(x, y):
    return np.mean(np.power(x-y, 2))

def logistic(w, x):
    return 1/(1+np.exp(-w[0] * x - w[1]))

def set_objective(x, y):
    def objective(w):
        return mse(logistic(w, x), y)
    return objective

def return_conv_params(conv_weight):
    conv_params = []
    for i in range(conv_weight.shape[0]):
        pwm = conv_weight[i].detach().cpu().numpy()
        if np.all(pwm.sum(axis=0) == 0):
            w = np.array([0,0])
        else:
            s, d = return_coef_for_normalization(pwm.T)
            f = set_objective(s, np.power(np.cumsum(d), 10))
            w = fmin(func=f, x0=np.array([1.0, 5.0]))
        conv_params.append(w)
    return conv_params

class TemplateModel(Module):
    def __init__(self, num_of_response, motif_path, window_size,
                 stride, pad_motif, reg_dim_expansion=1, both_direction = True,
                 initiate_conv_w_cdf = False):
        super(TemplateModel, self).__init__()

        #keep y dimension
        self.num_of_response = num_of_response

        #Reading motifs from a MEME file
        self.meme_file = MEME()
        self.motif_path = motif_path
        self.stride = stride
        kernels = self.meme_file.parse(motif_path)
        if not both_direction:
          kernels = kernels[::2]
        kernels = pad(kernels, (0,0,0,0,0,pad_motif), value=0)
        self.pad_motif = pad_motif

        #Extracting kernel dimensions for first convolutional layer definition
        self.out_channels, self.in_channels, self.kernel_size = kernels.shape

        #First layer convolution. Weights set to motifs and are fixed.
        self.conv_motif = Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride, bias=False)#, padding="same"
        self.conv_motif.weight = torch.nn.Parameter(kernels)
        self.conv_motif.weight.requires_grad = False
        self.conv_length = int((window_size - self.kernel_size) / stride) + 1

        self.conv_bias = Parameter(torch.empty(1, self.out_channels, 1, requires_grad = True))
        self.conv_scale  = Parameter(torch.empty(self.out_channels, requires_grad = True))
        if initiate_conv_w_cdf:
            params = return_conv_params(self.conv_motif.weight)
            with torch.no_grad():
                for i, p in enumerate(params):
                    self.conv_bias[:,i,:] = p[1]
                    self.conv_scale[i] = p[0]
        else:
            with torch.no_grad():
                torch.nn.init.ones_(self.conv_scale)
                torch.nn.init.zeros_(self.conv_bias)

        self.linreg = Linear(in_features=int(reg_dim_expansion*self.out_channels/2), out_features=num_of_response)


    def init_weights(self):
        for name, layer in self.named_children():
            if name == "conv_motif":
                pass
            elif name == "conv_bias":
                torch.nn.init.zeros_(self.conv_bias)
            elif name == "conv_scale":
                torch.nn.init.ones_(self.conv_scale)
            else:
                if isinstance(layer, Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
                elif isinstance(layer, Conv1d):
                    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "relu")
                else:
                    pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def unfreeze(self, layers = None, doall = False):
        if doall:
            for l in self.name_parameters():
                l.requires_grad = True
        elif layers is None:
            self.conv_motif.weight.requires_grad = True
        else:
            for n,l in self.named_parameters():
                if n.split(".")[0] in layers:
                    l.requires_grad = True

    def print(self):
        for n,l in self.named_parameters():
            print(n)

    def freeze(self, layers = None, doall = False):
        if doall:
            for l in self.parameters():
                l.requires_grad = False
        elif layers is None:
            self.conv_motif.weight.requires_grad = False
        else:
            for n,l in self.named_parameters():
                if n.split(".")[0] in layers:
                    l.requires_grad = False

    def motif_layer(self, x):
        x = self.conv_motif(x)
        return torch.einsum("bcl, c -> bcl", x, self.conv_scale) + self.conv_bias

    def forward(self, x):
        raise NotImplementedError

    def motif_ranks(self):
        with torch.no_grad():
            final_layer = self.linreg.weight.cpu().detach().numpy()
        ii = np.argsort(-final_layer, axis=1)
        names = self.meme_file.motif_names() + ([""] * self.pad_motif)
        return ii, np.array(names), final_layer

    def interactions(self):
        raise NotImplementedError
