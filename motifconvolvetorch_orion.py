#!/usr/bin/env python
import argparse
import numpy as np
import os
import numpy as np
from pysam import FastaFile
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import pandas as pd

torch.backends.cudnn.deterministic = True

def readbed(filename, up):
    data = pd.read_csv(filename, sep = "\t", header = None)
    chrs = data[0].to_numpy()
    start = data[1].to_numpy()
    end = data[2].to_numpy()
    if(data.shape[1] > 5): #get the strand
        print("Strand detected")
        up = int(np.floor(up))
        strand = data[5].to_numpy()
        #adjust the regions to acccount for strand and up
        start = start - (strand == "+") * up #[start[i]-up if strand[i]=="+" else start[i] for i in range(len(start))]
        end = end + (strand == "-") * up #[end[i]+up if strand[i]=="-" else end[i] for i in range(len(start))]
    return np.array(chrs), np.array(start, dtype = int), np.array(end, dtype = int)

def countlowercase(arr):
    return sum([1 for c in arr if c.islower()])

def stringstats(string):
    lowercaseratio = countlowercase(string)/len(string)
    string = string.upper()
    tmp = np.array(list(string))
    gccount = np.sum(np.logical_or(tmp == 'C', tmp == 'G'))/len(tmp)
    gcpattern = string.count("GC")/(len(tmp)-1)
    cgpattern = string.count("CG")/(len(tmp)-1)
    return np.array([gccount, gcpattern, cgpattern, lowercaseratio])

def returnonehot(string):
    string = string.upper()
    lookup = {'A':0, 'C':1, 'G':2, 'T':3}
    tmp = np.array(list(string))
    icol = np.where(tmp != 'N')[0]
    out = np.zeros((4,len(tmp)))
    irow = np.array([lookup[i] for i in tmp[icol]])
    if len(icol)>0:
        out[irow,icol] = 1
    return np.asarray(out)

def write_output(filename, mat, names):
    pd.DataFrame(mat, columns = names + ["GC_ratio", "GC_pattern", "CG_pattern", "Masked_ratio"]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)

def createmasks(len1, len2, mat):
    for i1, l1 in enumerate(len1):
        for i2, l2 in enumerate(len2):
            mat[i1, i2, int(np.abs(l1-l2+2)):, 0] = 0

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
        out = np.zeros((out_channels, width, height))

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

class SegmentData:
    def __init__(self, bed, batchsize, genome, windowsize, up, scale = 1):#Orion Addition
        self.chrs, self.starts, self.ends = readbed(bed, up)
        self.midpoints = np.asarray(np.ceil((self.starts + self.ends)/2),dtype=int)
        self.starts = self.midpoints - int(windowsize/2)
        self.ends = self.midpoints + int(windowsize/2)
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.windowsize = windowsize
        self.scale = scale#Orion Addition
        #self.padding = windowsize
        refs = self.seqs.references
        lengths = self.seqs.lengths
        self.limits = {refs[i]: lengths[i] for i in range(len(refs))}
        self.out = open("coordinatesUsed.bed", "w")

    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
      def pdf(x,loc,scale):#Orion Addition
        return 1./(torch.sqrt(2.*torch.tensor(torch.pi))*scale)*torch.exp(-torch.pow((x-loc)/scale, 2.)/2)#Orion Addition
      i1, i2 = i*self.batchsize, (i+1)*self.batchsize
      if i2 >= self.n: i2 = self.n
      batchsize = int(i2 - i1)
      height = self.windowsize #np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
      width = 4
      batch = np.zeros((batchsize, width, height)) 
      stats = np.empty((batchsize, 4))
      weight_tensor = torch.linspace(-5,5,height)#Orion Addition
      gaussian_weight = pdf(weight_tensor,0,self.scale)#Orion Addition
      gaussian_weight = gaussian_weight/torch.sum(gaussian_weight) * height #Normalizing the weights 
      for i, c, s, e in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2]):
          self.out.write(c+"\t"+str(s)+"\t"+str(e)+"\n")
          if s>0 and e<self.limits[c]:
              seg = self.seqs.fetch(c, s, e)
          else:
              seg = "N"*self.windowsize
          stats[i] = stringstats(seg)
          batch[i, :, :(e-s)] = returnonehot(seg)
      output = torch.mul(torch.from_numpy(batch),gaussian_weight)#Orion Addition
      return output, stats#Orion Addition
      

    def __del__(self):
        self.out.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--genome', required = True, help="fasta file for the genome")
    parser.add_argument('-m','--meme', required = True, help="meme file for the motifs")
    parser.add_argument('-b', '--bed', required = True, help="bed file for the peaks")
    parser.add_argument('--batch', default=128, type=int, help="batch size")
    parser.add_argument('-o','--out', required = True, help="output directory")
    parser.add_argument('-w','--window', default=240, type=int, help="window size")
    parser.add_argument('-u','--up', default=0, type=int, help="add upstream")
    parser.add_argument('-s','--scale', default=1, type=float, help="Standard deviation for Gaussian decay.")
    parser.add_argument('-t','--transform', choices =["none", "constant", "local"], default="none", help="transform probabilities to scoring matrix")
    parser.add_argument('-d','--mode', choices = ["max", "average"], default="max", help="Operation mode for the pooling layer (max/average)")
    args = parser.parse_args()

    motif = MEME()
    print(f"Reading motifs from {args.meme}")
    kernels = motif.parse(args.meme, args.transform)
    print(f"Reading peaks from {args.bed}")
    print(f"Genome: {args.genome}")
    print(f"Total window: {args.window+args.up}")
    print(f"Standard deviation(Gaussian decay): {args.scale}")

    segments = SegmentData(args.bed, args.batch, args.genome, int(args.window+args.up/2), args.up, args.scale)
    out = np.empty((segments.n, motif.nmotifs+4))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, motif.nmotifs:] = segments[i]
        tmp = F.conv1d(mat, kernels)
        if args.mode == "average":
            tmp = F.avg_pool1d(tmp, tmp.shape[2]).numpy()
        if args.mode == "max":
            tmp = F.max_pool1d(tmp, tmp.shape[2]).numpy()
        out[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)
    print(f"Writing the results to {args.out}")
    write_output(args.out, out, motif.names)

main()
