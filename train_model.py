from model import MaskNet
from utils import SeqDataset
import torch
from torch.optim import Adam
from adam import AdamL1, AdamMCP
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
parser.add_argument("--split_folder", required=True, help="Folder that stores train/val/test splits.")
parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--number_of_epochs", default=10, type=int, help="Number of epochs for training")
parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
parser.add_argument("--train_ratio", default=0.8, type=float, help="Train split ratio")
parser.add_argument("--validation_ratio", default=0.1, type=float, help="Validation split ratio")
parser.add_argument("--num_of_workers", default = 8, type=int, help="Number of workers for data loading")
parser.add_argument("--num_of_heads", default = 8, type=int, help="Number of heads for attention layer")
parser.add_argument("--penalty_param", default = 0.1, type=float, help="Hyperparameter for the regularization of the final layer")
parser.add_argument("--mcp_param", default = 3, type=float, help="Second hyperparameter for the mcp regularization of the final layer (first is --penalty_param)")
parser.add_argument("--penalty_type", default="l1", help="l1/mcp regularization")

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
penalty_param = args.penalty_param
penalty_type = args.penalty_type
mcp_beta = args.mcp_param
split_folder = args.split_folder

os.path.exists(model_output) or os.makedirs(model_output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskNet(8, meme_file, window_size, num_heads).to(device)


#mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
#mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
#mem = mem_params + mem_bufs # in bytes

model.init_weights()

#optimizer_seq = Adam([x[1] for x in model.named_parameters() if x[0] not in ["linreg.weight", "linreg.bias"]])
if penalty_type == "l1":
  optimizer = AdamL1([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
                       'penalty_hyper_param': 0},
                      {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
                       'penalty_hyper_param': penalty_param}])
elif penalty_type == "mcp":
  optimizer = AdamMCP([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
                        'penalty_hyper_param': 0},
                       {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
                        'penalty_hyper_param': penalty_param,
                        'b': 3}])
else:
  raise ValueError(f"Invalid penalty type{penalty_type}.")

#optimizer = Adam(model.parameters())
#optimizer_linreg = ProxSGD([model.linreg.weight, model.linreg.bias], mu = 100000000)
#optimizer_linreg = ProxSGD(model.parameters(), mu = 0.0001)

dataset = SeqDataset(signal_file, sequence_file)

num_points = len(dataset)

#train_size = int(train_ratio * num_points)
#validation_size = int(validation_ratio * num_points)
#test_size = num_points - train_size - validation_size
#
#train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
#
train_indices = np.load(os.path.join(split_folder, "train_split.npy"))
validation_indices = np.load(os.path.join(split_folder, "validation_split.npy"))
test_indices = np.load(os.path.join(split_folder, "test_split.npy"))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

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
  trainingloss = [] 
  validationloss = []
  model.train()
  for x, y in train_dataloader:
    x, y = x.to(device), y.to(device)

    pred = model(x)
    currloss = loss(pred, y)# + l1_param * model.linreg.weight.abs().sum()

    optimizer.zero_grad()

    currloss.backward()

    optimizer.step()

    trainingloss.append(currloss.cpu().detach().numpy())

  with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      validationloss.append(loss(pred, y).cpu().detach().numpy())

    num_zero = [torch.sum(l.weight == 0).cpu() for l in model.linreg]
    stats['train_average_loss'].append(np.mean(trainingloss))
    stats['validation_average_loss'].append(np.mean(validationloss))
    print(f"Epoch {e+1}: Average loss (training) = {stats['train_average_loss'][-1]}")
    print(f"Epoch {e+1}: Average loss (validation) = {stats['validation_average_loss'][-1]}")
    print(f"Epoch {e+1}: Min. number of zero = {np.min(num_zero)}, Max. number of zero = {np.max(num_zero)}")
    model.save_model(os.path.join(model_output, f"model.{e}"))

    save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))

testloss = []
with torch.no_grad():
  model.eval()
  for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    pred = model(x)
    testloss.append(loss(pred, y).cpu().detach().numpy())
  stats['test_average_loss'] = np.mean(testloss)
  print(f"Average loss (test) = {stats['test_average_loss']}")

stats['completed'] = True

save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))
