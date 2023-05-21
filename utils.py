import torch
import pandas as pd
import numpy as np
#from pysam import FastaFile
#from Bio import SeqIO
#from pyfaidx import Fasta

import matplotlib
import matplotlib.pyplot as plt

def readbed(filename):
    data = pd.read_csv(filename, sep = "\t", header = None)
    return data.iloc[:,0].to_numpy(), np.asarray(np.ceil(data.iloc[:,[1,2]].sum(axis=1)/2), dtype = int)

def returnonehot(string):
    string = string.upper()
    lookup = {'A':0, 'C':1, 'G':2, 'T':3}
    tmp = np.array(list(string))
    icol = np.where(tmp != 'N')[0]
    out = np.zeros((4,len(tmp)), dtype = np.float32)
    irow = np.array([lookup[i] for i in tmp[icol]])

    if len(icol)>0:
        out[irow,icol] = 1

    return out

class MEME():
    def __init__(self):
        self.version = 0
        self.alphabet = ""
        self.strands = ""
        self.headers = []
        self.background = []
        self.names = []
        self.nmotifs = 0

    def parse(self, text, transform):
        with open(text,'r') as file:
            data = file.read()
        data = data.split("\n\n")
        data = data[:-1]

        offset_metadata = 4
        self.nmotifs = len(data) - offset_metadata
        self.version = int(data[0].split(' ')[-1])
        self.alphabet = data[1][10:].strip()
        self.strands = data[2][9:].strip()
        self.background = np.array(data[3].split('\n')[1].split(' ')[1::2],dtype=float)

        out_channels = self.nmotifs * 2

        lens = np.array([len(i.split('\n')[2:]) for i in data[offset_metadata:]])
        height = np.max(lens)
        maximumpadding = height - np.min(lens)
        width = len(self.alphabet)
        out = np.zeros((out_channels, width, height), dtype = np.float32)

        data = data[offset_metadata:]
        for k, i in enumerate(data):
            tmp = i.split('\n')
            self.names.append(tmp[0].split()[-1])
            self.headers.append('\n'.join(tmp[:2]))
            kernel = np.array([j.split() for j in tmp[2:]],dtype=float).T
            if (transform == "constant"):
                bg=np.repeat(0.25,width).reshape(1,width)
            if (transform == "local"):
                bg=np.average(kernel,0).reshape(1,width)
            if (transform != "none"):
                offset=np.min(kernel[kernel>0])
                kernel=np.log((kernel+offset)/bg)
            kernel_start = int((height - kernel.shape[1])/2)
            kernel_end = kernel_start + kernel.shape[1]
            out[2*k  , :, kernel_start:kernel_end] = kernel
            out[2*k+1, :, kernel_start:kernel_end] = kernel[::-1, ::-1]
        return torch.from_numpy(out)

    def motif_names(self):
        return self.names

class SeqDataset(torch.utils.data.Dataset):

  def __init__(self, bedGraph_path, sequence_path):
    self.sequences = pd.read_csv(sequence_path, header = None, sep = "\t", index_col = 0)
    self.atacsignal = pd.read_csv(bedGraph_path, header = 0, sep = "\t")

  def __len__(self):
    return self.atacsignal.shape[0]

  def __getitem__(self, index):
    seq_name = self.atacsignal.iloc[index, 0]
    sequence = self.sequences.loc[seq_name, 1]
    return returnonehot(sequence), self.atacsignal.iloc[index, 1:].to_numpy(dtype=np.float32)

  def cell_types(self):
    return self.atacsignal.columns.to_numpy()[1:]

  def number_of_cell_types(self):
    return len(self.atacsignal.columns.to_numpy()[1:])

#==== viz_sequence
#{{{

def ic_scale(pwm,background):
    odds_ratio = ((pwm+0.001)/(1.004))/(background[None,:])
    ic = ((np.log((pwm+0.001)/(1.004))/np.log(2))*pwm -\
            (np.log(background)*background/np.log(2))[None,:])
    return pwm*(np.sum(ic,axis=1)[:,None])


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 ylabel="",
                 ylim=None):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))

    if ylim is not None:
        # Use user-specified y-axis limits
        min_neg_height, max_pos_height = ylim
        assert min_neg_height <= 0
        assert max_pos_height >= 0

    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ylabel="",
                 ylim=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight,
        ylabel=ylabel,
        ylim=ylim)
    plt.show()
#}}}
