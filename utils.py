import torch
import pandas as pd
import numpy as np
#from pysam import FastaFile
#from Bio import SeqIO
#from pyfaidx import Fasta

def init_dist(dmin, dmax, dp, weights, probs):
    out = np.zeros(int(np.round((dmax-dmin)/dp)+1))
    ii = np.array(np.round((weights-dmin)/dp), dtype=int)
    for i in range(len(probs)):
        out[ii[i]] = out[ii[i]] + probs[i]
    return out

def score_dist(pwm, nucleotide_prob=None, gran=None, size=1000):
    if nucleotide_prob is None:
        nucleotide_prob = np.ones(4)/4
    if gran is None:
        if size is None:
            raise ValueError("provide either gran or size. Both missing.")
        gran = (np.max(pwm) - np.min(pwm))/(size - 1)
    pwm = np.round(pwm/gran)*gran
    pwm_max, pwm_min = pwm.max(axis=1), pwm.min(axis=1)
    distribution = init_dist(pwm_min[0], pwm_max[0], gran, pwm[0], nucleotide_prob[0])
    for i in range(1, pwm.shape[0]):
        kernel = init_dist(pwm_min[i], pwm_max[i], gran, pwm[i], nucleotide_prob[i])
        distribution = np.convolve(distribution, kernel)
    support_min = pwm_min.sum()
    ii = np.where(distribution > 0)[0]
    support = support_min + (ii) * gran
    return support, distribution[ii]

def return_coef_for_normalization(pwm, nucleotide_prob=None, gran=None, size=1000):
    pwm = pwm[pwm.sum(axis=1) != 0, :]
    prob = np.exp(pwm) / np.exp(pwm).sum(axis=1).reshape(-1,1)
    return score_dist(pwm, prob, gran, size)

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
    def __init__(self, do_log = False):
        self.version = 0
        self.alphabet = ""
        self.strands = ""
        self.headers = []
        self.background = []
        self.names = []
        self.nmotifs = 0
        self.do_log = do_log
        self.epsilon = 1e-9

    def parse(self, text):
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
            kernel_start = int((height - kernel.shape[1])/2)
            kernel_end = kernel_start + kernel.shape[1]
            out[2*k  , :, kernel_start:kernel_end] = kernel
            out[2*k+1, :, kernel_start:kernel_end] = kernel[::-1, ::-1]
        if self.do_log:
            out = np.log(out + self.epsilon)
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
