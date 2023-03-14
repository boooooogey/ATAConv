import torch
import pandas as pd
import numpy as np
#from pysam import FastaFile
#from Bio import SeqIO
#from pyfaidx import Fasta

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
