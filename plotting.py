"""
Useful plotting functions for epigenomic models.
"""
from typing import Tuple
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from torch import Tensor

import seaborn
import torch
import numpy as np
import matplotlib.colors as clr

def annotate_seq(one_hot: Tensor.float, signal: Tensor.float)-> Axes:
    """Highlights with the given signal

    Args:
        one_hot (Tensor.float): _description_
        signal (Tensor.float): _description_

    Returns:
        Axes: _description_
    """
    sequence = one_hot_to_sequence(one_hot)
    signal = map_conv_to_seq(signal, sequence.size).reshape(1,-1)
    return sequence, signal
    #return plot_sequence(sequence,
    #                     signal)

def one_hot_to_sequence(one_hot: Tensor) -> ndarray:
    """Takes one hot encoded sequence and return the sequence.

    Args:
        one_hot (Tensor): one hot encoding of a sequence

    Returns:
        ndarray: DNA sequence
    """
    lookup = {0: "A", 1:"C", 2:"G", 3:"T"}
    indices = one_hot.argmax(axis=0)
    return np.array([[lookup[i.item()] for i in indices]])

def map_conv_to_seq(signal: Tensor.float, length: int) -> Tensor.float:
    """Map signal (convoluted) back onto the sequence using tranpose convolution.

    Args:
        signal (Tensor.float): Signal to be mapped.
        length (int): length of the sequence. 

    Returns:
        Tensor.float: Signal mapped back to the sequence 
    """
    kernel_size = length - signal.nelement() + 1
    weights = torch.ones(1, 1, kernel_size) / kernel_size
    return torch.nn.functional.conv_transpose1d(signal.reshape(1,1,-1), weights)[0,0]

def plot_sequence(sequence: str,
                  signal: Tensor.float,
                  figsize: Tuple[float, float] = None,
                  fontsize: int = 12,
                  vminmax: Tuple[float, float] = (None, None)) -> Axes:
    """Plots given DNA sequence as heatmap and highlights with given signal.

    Args:
        sequence (str): DNA sequence
        signal (Tensor.float): signal mapped onto the given sequence. Example: TF binding site
        convolution
        figsize (Tuple[float, float]): figure size for the heatmap
        fontsize (int): The font size

    Returns:
        Returns Axes object for the heatmap
    """
    if vminmax[0] is None:
        norm = clr.Normalize(vmin = signal.min(), vmax = signal.max())
    else:
        norm = clr.Normalize(vmin = vminmax[0], vmax = vminmax[1])
    if figsize is None:
        pt = (fontsize / 1.5 * 4) / 100
        figsize = (pt * sequence.size, pt)
    _, ax = plt.subplots(figsize = figsize)
    ax = seaborn.heatmap(signal,
                         annot=sequence,
                         fmt="",
                         cbar=False,
                         xticklabels="",
                         yticklabels="",
                         norm=norm,
                         cmap="Reds",
                         annot_kws={"size":fontsize})
    return ax
