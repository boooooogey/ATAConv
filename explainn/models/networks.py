# =============================================================================
# IMPORTS
# =============================================================================
from collections import OrderedDict

import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch.nn.modules.activation as activation

# added
import sys
from utils import MEME

# =============================================================================
# MODEL CLASSES
# =============================================================================
class exp_activation_module(nn.Module):
    """
    Exponential activation function
    (from Koo, 2021, PMID: 34322657)
    """
    def __init__(self):
        super(exp_activation_module, self).__init__()
    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)

class ConvNetDeep(nn.Module):
    """
    CNN with 3 conv layers;
    Inspired by Basset (PMID: 27197224)
    """
    def __init__(self, num_classes, weight_path=None):
        """
        initialize the model
        designed for input sequences of length 200 bp

        :param num_classes: int, number of outputs
        :param weight_path: string, path to the file with model weights
        """
        super(ConvNetDeep, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4, 100, 19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU()
        self.mp1 = nn.MaxPool1d(3, 3)

        # Block 2 :
        self.c2 = nn.Conv1d(100, 200, 7)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(3, 3)

        # Block 3 :
        self.c3 = nn.Conv1d(200, 200, 4)
        self.bn3 = nn.BatchNorm1d(200)
        self.rl3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(3, 3)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(1000, 1000)  # 1000 for 200 input size
        self.bn4 = nn.BatchNorm1d(1000, 1e-05, 0.1, True)
        self.rl4 = activation.ReLU()
        self.dr4 = nn.Dropout(0.3)

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000, 1e-05, 0.1, True)
        self.rl5 = activation.ReLU()
        self.dr5 = nn.Dropout(0.3)

        # Block 6 :4Fully connected 3
        self.d6 = nn.Linear(1000, num_classes)
        # self.sig = activation.Sigmoid()

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x, embeddings=False):
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        x = self.rl1(self.bn1(self.c1(x)))

        activations = x
        x = self.mp1(x)
        x = self.mp2(self.rl2(self.bn2(self.c2(x))))
        em = self.mp3(self.rl3(self.bn3(self.c3(x))))
        o = torch.flatten(em, start_dim=1)
        o = self.dr4(self.rl4(self.bn4(self.d4(o))))
        o = self.dr5(self.rl5(self.bn5(self.d5(o))))
        o = self.d6(o)

        activations, act_index = torch.max(activations, dim=2)

        if embeddings: return (o, activations, act_index, em)
        return o

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)


#################################################################################
class ConvNetShallow(nn.Module):
    """
    Shallow CNN model with 1 convolutional layer
    """
    def __init__(self, num_classes, weight_path=None):
        """
        initialize the model
        designed for input sequences of length 200 bp

        :param num_classes: int, number of output classes
        :param weight_path: string, path to the file with model weights
        """
        super(ConvNetShallow, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4, 100, 19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU()
        self.mp1 = nn.MaxPool1d(7, 7)

        # Block 4 : Fully Connected 1 :
        self.d2 = nn.Linear(2600, 1000)  # 1000 for 200 input size
        self.bn2 = nn.BatchNorm1d(1000, 1e-05, 0.1, True)
        self.rl2 = activation.ReLU()
        self.dr2 = nn.Dropout(0.3)

        # Block 5 : Fully Connected 2 :
        self.d3 = nn.Linear(1000, 1000)
        self.bn3 = nn.BatchNorm1d(1000, 1e-05, 0.1, True)
        self.rl3 = activation.ReLU()
        self.dr3 = nn.Dropout(0.3)

        # Block 6 :4Fully connected 3
        self.d4 = nn.Linear(1000, num_classes)
        # self.sig = activation.Sigmoid()

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x, embeddings=False):
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        x = self.rl1(self.bn1(self.c1(x)))

        activations = x
        em = self.mp1(x)
        o = torch.flatten(em, start_dim=1)
        o = self.dr2(self.rl2(self.bn2(self.d2(o))))
        o = self.dr3(self.rl3(self.bn3(self.d3(o))))
        o = self.d4(o)

        activations, act_index = torch.max(activations, dim=2)

        if embeddings: return (o, activations, act_index, em)
        return o

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)


#################################################################################
class DanQ(nn.Module):
    """
    Implementation of DanQ architecture
    (PMID: 27084946)
    """
    def __init__(self, num_classes, weight_path=None):
        """
        initialize the model
        designed for input sequences of length 200 bp

        :param num_classes: int, number of output classes
        :param weight_path: string, path to the file with model weights
        """
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(13 * 640, 925)
        self.Linear2 = nn.Linear(925, num_classes)

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 13 * 640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)


class ExplaiNN(nn.Module):
    """
    Class for the ExplaiNN model
    """
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None):
        """
        initialize the model

        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN, self).__init__()

        #Reading motifs from a MEME file
        self.meme_file = MEME()
        self.motif_path = weight_path
        self.kernels = self.meme_file.parse(weight_path, "none")
        self.weights = torch.nn.Parameter(self.kernels)

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                exp_activation_module(),
                nn.MaxPool1d(input_length - (filter_size-1)),
                nn.Flatten())
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                exp_activation_module(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                exp_activation_module(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
            self.linears[0].weight = self.weights
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                exp_activation_module(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU())

            self.linears_bg = nn.ModuleList([nn.Sequential(nn.Dropout(0.3),
                                                           nn.Conv1d(in_channels=100 * num_cnns,
                                                                     out_channels=100 * num_cnns, kernel_size=1,
                                                                     groups=num_cnns),
                                                           nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                                                           nn.ReLU()) for i in range(num_fc - 2)])

            self.last_linear = nn.Sequential(nn.Dropout(0.3),
                                             nn.Conv1d(in_channels=100 * num_cnns, out_channels=1 * num_cnns,
                                                       kernel_size=1,
                                                       groups=num_cnns),
                                             nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                                             nn.ReLU(),
                                             nn.Flatten())

        self.final = nn.Linear(num_cnns, num_classes)
        self.num_cnns = num_cnns
        self.num_fc = num_fc

    def forward(self, x):
        x = x.repeat(1, self.num_cnns, 1)

        if self.num_fc <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)

        out = self.final(outs)

        return out

    def init_weights(self):
        pass_one = False
        for i in range(self.linears.__len__()):
            print(self.linears[i])
            if isinstance(self.linears[i], nn.Linear):                                                                                                                      
                torch.nn.init.kaiming_uniform_(self.linears[i].weight, nonlinearity = "relu")                                                                              
            elif isinstance(self.linears[i], nn.Conv1d) and pass_one == True:                                                                                                                    
                torch.nn.init.kaiming_normal_(self.linears[i].weight, nonlinearity = "relu")                                                                               
            if isinstance(self.linears[i], nn.Conv1d):                                                                                                                    
                pass_one = True                                                                          
            else:
                pass

class SingleLayer(nn.Module):
    """
    Single output layer;
    Useful when predicting on the outputs of other models
    """

    def __init__(self, num_inputs, num_classes, weight_path=None):
        """
        initialize the layer

        :param num_inputs: int, number of input features
        :param num_classes: int, number of outputs
        :param weight_path: string, path to the file with weights
        """
        super(SingleLayer, self).__init__()

        self.final = nn.Linear(num_inputs, num_classes)

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x):

        out = self.final(x.float())

        return out

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)


class NAMLayer(nn.Module):
    """
    Same as the linear class, but NAM style
    (extra FC layers before the final output)
    """
    def __init__(self, num_inputs, num_hidden, num_classes, weight_path=None):
        """
        initialize the model

        :param num_inputs: int, number of inputs
        :param num_hidden: int, size of the FC hidden layer
        :param num_classes: int, number of output classes
        :param weight_path: string, path to file with weights
        """
        super(NAMLayer, self).__init__()

        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=1 * num_inputs, out_channels=num_hidden * num_inputs,
                      kernel_size=1,
                      groups=num_inputs),
            nn.BatchNorm1d(num_hidden * num_inputs, 1e-05, 0.1, True),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Conv1d(in_channels=num_hidden * num_inputs, out_channels=num_hidden * num_inputs,
            #          kernel_size=1,
            #          groups=num_inputs),
            # nn.BatchNorm1d(num_hidden*num_inputs,1e-05,0.1,True),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Conv1d(in_channels=num_hidden * num_inputs, out_channels=1 * num_inputs,
                      kernel_size=1,
                      groups=num_inputs),
            nn.Flatten()
        )

        self.final = nn.Linear(num_inputs, num_classes)

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x):

        x = x.unsqueeze(-1)
        xout = self.linear(x)
        out = self.final(xout)

        return out

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)

class DeepSTARR(nn.Module):
    """
    Model taken from Almeida et al
    (PMID: 35551305)
    """
    def __init__(self, weight_path=None):
        super(DeepSTARR, self).__init__()
        self.convol = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=7,
                      padding=int((7 - 1) / 2)),  # same padding
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=256, out_channels=60, kernel_size=3,
                      padding=int((3 - 1) / 2)),  # same padding
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5,
                      padding=int((5 - 1) / 2)),  # same padding
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=60, out_channels=120, kernel_size=3,
                      padding=int((3 - 1) / 2)),  # same padding
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(1800, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x):

        o = self.linear(self.convol(x))

        return o

    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1:
                if v.shape[-1] == 1:
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)

class NonStrandSpecific(nn.Module):
    '''
    Adapted from Selene:
    https://github.com/FunctionLab/selene/blob/master/selene_sdk/utils/non_strand_specific_module.py

    A torch.nn.Module that wraps a user-specified model architecture if the
    architecture does not need to account for sequence strand-specificity.

    Parameters
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}, optional
        Default is 'mean'. NonStrandSpecific will pass the input and the
        reverse-complement of the input into `model`. The mode specifies
        whether we should output the mean or max of the predictions as
        the non-strand specific prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}
        How to handle outputting a non-strand specific prediction.
    '''

    def __init__(self, model):
        super(NonStrandSpecific, self).__init__()

        self.model = model

    def forward(self, input):
        reverse_input = None
        reverse_input = _flip(_flip(input, 1), 2)

        output = self.model.forward(input)
        output_from_rev = self.model.forward(reverse_input)

        return((output + output_from_rev) / 2)