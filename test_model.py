import pickle
import torch
import numpy as np
import argparse
import os
from model import MaskNet
from utils import SeqDataset

parser = argparse.ArgumentParser()
parser.add_argument("--meme_file", required=True, help="Path to the meme file that stores PWMs")
parser.add_argument("--atac_file", required=True, help="Path to the file that stores ATAC signal")
parser.add_argument("--sequences", required=True, help="Path to the file that stores sequences")
parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
parser.add_argument("--model", required=True, help="Model file")
parser.add_argument("--split_folder", required=True, help="Folder that stores train/val/test splits.")
parser.add_argument("--num_of_heads", default = 8, type=int, help="Number of heads for attention layer")
parser.add_argument("--num_of_workers", default = 8, type=int, help="Number of workers for data loading")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

meme_file = args.meme_file # "../motif-Convo-orion/local/Test.meme"
window_size = args.window_size # 300
signal_file = args.atac_file # "local/ATACseqSignal.first10k.txt"
num_heads = args.num_of_heads # 8
sequence_file = args.sequences # "local/sequences.list"
num_of_workers = args.num_of_workers # 8
model_name = args.model
split_folder = args.split_folder
batch_size = args.batch_size # 254

model = MaskNet(8, meme_file, window_size, num_heads).to(device)

model.load_model(model_name)

dataset = SeqDataset(signal_file, sequence_file)

test_indices = np.load(os.path.join(split_folder, "test_split.npy"))
test_dataset = torch.utils.data.Subset(dataset, test_indices)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

loss = torch.nn.MSELoss(reduction="mean")

testloss = []
with torch.no_grad():
  model.eval()
  for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    pred = model(x)
    testloss.append(loss(pred, y).cpu().detach().numpy())
  testl = np.mean(testloss)
  print(f"Average loss (test) = {testl}")

