import torch
import pandas as pd
import numpy as np
from pysam import FastaFile

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

    return np.asarray(out)

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
            out[2*k  , :, :kernel.shape[1]] = kernel
            out[2*k+1, :, :kernel.shape[1]] = kernel[::-1, ::-1]
        return torch.from_numpy(out)

class SeqDataset(torch.utils.data.Dataset):

  def __init__(self, bed, genome, window_size):
    self.chrom_list, self.midpoint_list = readbed(bed)
    self.window_offset = int(window_size/2)
    self.window_size = 2*self.window_offset
    self.genome = FastaFile(genome)

    chrom_references = self.genome.references
    chrom_sizes = self.genome.lengths

    self.limits = {chrom_references[i]: chrom_sizes[i] for i in range(len(chrom_references))}

  def __len__(self):
    return len(self.midpoint_list)

  def __getitem__(self, index):
    sequence = self.genome.fetch(self.chrom_list[index], self.midpoint_list[index] - self.window_offset, self.midpoint_list[index] + self.window_offset)
    return returnonehot(sequence)
