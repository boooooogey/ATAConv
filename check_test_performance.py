from model_simple_attention_poolingdiag_positional_embedding_symmetric_sigmoidconv import SimpleNet
from utils import SeqDataset
import torch
import os
from IPython import embed
import pickle
import numpy as np
import argparse

def save_to_pickle(data, filepath):
  with open(filepath, "wb") as file:
    pickle.dump(data, file)

parser = argparse.ArgumentParser()
parser.add_argument("--meme_file", required=True, help="Path to the meme file that stores PWMs")
parser.add_argument("--atac_file", required=True, help="Path to the file that stores ATAC signal")
parser.add_argument("--sequences", required=True, help="Path to the file that stores sequences")
parser.add_argument("--split", required=True, help="subset of the data.")
parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--number_of_epochs", default=10, type=int, help="Number of epochs for training")
parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
parser.add_argument("--num_of_workers", default = 8, type=int, help="Number of workers for data loading")
parser.add_argument("--model", required=True, help="Model")
parser.add_argument("--test", required=True, help="File save")

args = parser.parse_args()

window_size = args.window_size # 300
number_of_epochs = args.number_of_epochs # 2
meme_file = args.meme_file # "../motif-Convo-orion/local/Test.meme"
signal_file = args.atac_file # "local/ATACseqSignal.first10k.txt"
sequence_file = args.sequences # "local/sequences.list"
batch_size = args.batch_size # 254
num_of_workers = args.num_of_workers # 8
split = args.split
model_name = args.model
test_file = args.test

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device_name)

model = SimpleNet(8, meme_file, window_size).to(device)

model.load_model(model_name)

model.eval()

dataset = SeqDataset(signal_file, sequence_file)

test_indices = np.load(split)

test_dataset = torch.utils.data.Subset(dataset, test_indices)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

loss = torch.nn.MSELoss(reduction="mean")
k = 1

testloss = []
testy = []
testpred = [] 
with torch.no_grad():
  for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    pred, _ = model(x)
    testloss.append(loss(pred, y).cpu().detach().numpy())
    testpred.append(pred.cpu().detach().numpy())
    testy.append(y.cpu().detach().numpy())
    #testresults.append(torch.hstack([pred[:,k].view(-1,1), y[:,k].view(-1,1)]).cpu().detach().numpy())
  aveloss = np.mean(testloss)
  print(f"Average loss (test) = {aveloss}")

testy = np.vstack(testy)
testpred = np.vstack(testpred)
testresults = np.zeros((2,testy.shape[0], testy.shape[1]))
testresults[0,:,:] = testpred
testresults[1,:,:] = testy
#testresults = np.stack(testresults, axis=2).reshape(8, -1, 2)
np.save(test_file, testresults)

