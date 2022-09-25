from model import MaskNet
from utils import SeqDataset
import torch
from torch.optim import Adam
from adam import AdamL1
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
parser.add_argument("--model_output", required=True, help="Directory to store model parameters")
parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--number_of_epochs", default=10, type=int, help="Number of epochs for training")
parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
parser.add_argument("--train_ratio", default=0.8, type=float, help="Train split ratio")
parser.add_argument("--validation_ratio", default=0.1, type=float, help="Validation split ratio")
parser.add_argument("--num_of_workers", default = 8, type=int, help="Number of workers for data loading")
parser.add_argument("--num_of_heads", default = 8, type=int, help="Number of heads for attention layer")

args = parser.parse_args()

window_size = args.window_size # 300
number_of_epochs = args.number_of_epochs # 2
meme_file = args.meme_file # "../motif-Convo-orion/local/Test.meme"
signal_file = args.atac_file # "local/ATACseqSignal.first10k.txt"
sequence_file = args.sequences # "local/sequences.list"
batch_size = args.batch_size # 254
model_output = args.model_output # "local/test/"
train_ratio = args.train_ratio # 0.8
validation_ratio = args.validation_ratio # 0.1
num_of_workers = args.num_of_workers # 8
num_heads = args.num_of_heads # 8

os.path.exists(model_output) or os.makedirs(model_output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskNet(8, meme_file, window_size, num_heads).to(device)

l1_param = [0 for i in model.parameters()]
l1_param[len(l1_param)-2] = 1000

#mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
#mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
#mem = mem_params + mem_bufs # in bytes

model.init_weights()

#optimizer_seq = Adam([x[1] for x in model.named_parameters() if x[0] not in ["linreg.weight", "linreg.bias"]])
#optimizer = AdamL1(model.parameters(), l1_param)
optimizer = Adam(model.parameters())
#optimizer_linreg = ProxSGD([model.linreg.weight, model.linreg.bias], mu = 100000000)
#optimizer_linreg = ProxSGD(model.parameters(), mu = 0.0001)

dataset = SeqDataset(signal_file, sequence_file)
num_points = len(dataset)
train_size = int(train_ratio * num_points)
validation_size = int(validation_ratio * num_points)
test_size = num_points - train_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

np.save(os.path.join(model_output, "train_split.npy"), train_dataset.indices)
np.save(os.path.join(model_output, "validation_split.npy"), validation_dataset.indices)
np.save(os.path.join(model_output, "test_split.npy"), test_dataset.indices)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_of_workers)
number_of_step = len(train_dataset)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

loss = torch.nn.MSELoss(reduction="mean")
#embed()

stats = {
    'train_average_loss' : [],
    'validation_average_loss' : [],
    'test_average_loss' : 0,
    'completed' : False
    }

for e in range(number_of_epochs):
  trainingloss = 0
  validationloss = 0
  model.train()
  for x, y in train_dataloader:
    x, y = x.to(device), y.to(device)

    pred = model(x)
    currloss = loss(pred, y)# + l1_param * model.linreg.weight.abs().sum()
    #print(currloss)
    trainingloss += currloss

    optimizer.zero_grad()
    #optimizer_seq.zero_grad()
    #optimizer_linreg.zero_grad()

    currloss.backward()

    optimizer.step()
    #optimizer_linreg.step()
    #optimizer_seq.step()

  with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      validationloss += loss(pred, y)
    print(f"Epoch {e+1}: Average loss (training) = {trainingloss/number_of_step}")
    print(f"Epoch {e+1}: Average loss (validation) = {validationloss/number_of_step}")
    model.save_model(os.path.join(model_output, f"model.{e}"))

    stats['train_average_loss'].append(trainingloss.cpu().detach().numpy()/len(train_dataset))
    stats['validation_average_loss'].append(validationloss.cpu().detach().numpy()/len(validation_dataset))
    save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))

testloss = 0
with torch.no_grad():
  model.eval()
  for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    pred = model(x)
    testloss += loss(pred, y)
  print(f"Average loss (test) = {testloss/number_of_step}")
  stats['test_average_loss'] = testloss.cpu().detach().numpy()/len(test_dataset)

stats['completed'] = True

save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))
