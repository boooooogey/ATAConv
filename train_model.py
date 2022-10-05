from model import MaskNet
from utils import SeqDataset
import torch
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau   
from adam import AdamL1, AdamMCP
import os
from IPython import embed
import pickle
import numpy as np
import argparse

# add
import sys
import os 
from aitac import *

PREFIX="/net/talisker/home/benos/mae117/collaborations/atac_convolutions/data"

def save_to_pickle(data, filepath):
  with open(filepath, "wb") as file:
    pickle.dump(data, file)

parser = argparse.ArgumentParser()
parser.add_argument("--meme_file", default=f"{PREFIX}/cisBP_mouse.meme", help="Path to the meme file that stores PWMs")
parser.add_argument("--atac_file", default=f"{PREFIX}/lineageImmgenDataZ.txt", help="Path to the file that stores ATAC signal")
parser.add_argument("--sequences",default=f"{PREFIX}/sequences.list", help="Path to the file that stores sequences")
parser.add_argument("--model_output",default=f"{PREFIX}/outputs", help="Directory to store model parameters")
parser.add_argument("--split_folder",default=f"{PREFIX}/splits", help="Folder that stores train/val/test splits.")
parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--number_of_epochs", default=10000, type=int, help="Number of epochs for training")
parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
parser.add_argument("--num_of_workers", default = 8, type=int, help="Number of workers for data loading")

parser.add_argument("--model", default = f"{PREFIX}/outputs/model.371", help="Start from the given state.")
#parser.add_argument("--model", default = None, help="Start from the given state.")

parser.add_argument("--num_of_heads", default = 8, type=int, help="Number of heads for attention layer")
parser.add_argument("--penalty_param", default = 0.1, type=float, help="Hyperparameter for the regularization of the final layer")
parser.add_argument("--mcp_param", default = 3, type=float, help="Second hyperparameter for the mcp regularization of the final layer (first is --penalty_param)")
parser.add_argument("--penalty_type", default="l1", help="l1/mcp regularization")

parser.add_argument("--lr", default = 0.001, type=float, help="Learning rate")                                                                               
#parser.add_argument("--lr", default = None, type=float, help="Learning rate")                                                                               

parser.add_argument("--response_num", default = 8, type=int, help="number of signals to predict")                                                          

args = parser.parse_args()

window_size = args.window_size # 300
number_of_epochs = args.number_of_epochs # 2
meme_file = args.meme_file # "../motif-Convo-orion/local/Test.meme"
signal_file = args.atac_file # "local/ATACseqSignal.first10k.txt"
sequence_file = args.sequences # "local/sequences.list"
batch_size = args.batch_size # 254
model_output = args.model_output # "local/test/"
num_of_workers = args.num_of_workers # 8
num_heads = args.num_of_heads # 8
penalty_param = args.penalty_param
penalty_type = args.penalty_type
mcp_beta = args.mcp_param
split_folder = args.split_folder
model_name = args.model
learning_rate = args.lr
response_dim = args.response_num

# Added to recapitulate
num_filters=300

if model_name is None:                                                                                                                                     
    model_i = 0                                                                                                                                            
else:                                                                                                                                                      
    model_i = int(model_name.split(".")[-1])                                                                                                               


grad_path = os.path.join(model_output, "grad")                             
                                                                           
os.path.exists(model_output) or os.makedirs(model_output)                  
os.path.exists(grad_path) or os.makedirs(grad_path)                        

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device_name)

#model = MaskNet(8, meme_file, window_size, num_heads).to(device)
model = ConvNet(num_classes=response_dim, num_filters=num_filters).to(device)


#mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
#mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
#mem = mem_params + mem_bufs # in bytes

if model_name is None:                                                                                                                                     
  pass
  #model.init_weights()                                                                                                                                   
else:                                                                                                                                                      
    model.load_model(model_name)

# OLD IMPLEMENTATION ===
#optimizer_seq = Adam([x[1] for x in model.named_parameters() if x[0] not in ["linreg.weight", "linreg.bias"]])
# if penalty_param != 0:
#   if penalty_type == "l1":
#     optimizer = AdamL1([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
#                          'penalty_hyper_param': 0},
#                         {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
#                          'penalty_hyper_param': penalty_param}])
#   elif penalty_type == "mcp":
#     optimizer = AdamMCP([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
#                           'penalty_hyper_param': 0},
#                          {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
#                           'penalty_hyper_param': penalty_param,
#                           'b': 3}])
#   else:
#     raise ValueError(f"Invalid penalty type{penalty_type}.")
# else:
#   optimizer = AdamW(model.parameters())

# scheduler = ExponentialLR(optimizer, gamma=0.9) 
# OLD IMPLEMENTATION ====

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

if os.path.exists(os.path.join(model_output, "stats.pkl")):                                                                                                
    with open(os.path.join(model_output, "stats.pkl"), "rb") as file:                                                                                      
        stats = pickle.load(file)                                                                                                                          
    stats['train_average_loss'] = stats['train_average_loss'][:model_i+1]                                                                                  
    stats['validation_average_loss'] = stats['validation_average_loss'][:model_i+1]                                                                        
    stats['lr'] = stats['lr'][:model_i+1]                                                                                                                  
else:                                                                                                                                                      
    stats = {                                                                                                                                              
        'train_average_loss' : [],                                                                                                                         
        'validation_average_loss' : [],                                                                                                                    
        'test_average_loss' : 0,                                                                                                                           
        'completed' : False                                                                                                                                
        }                                                                                                                                                  

if learning_rate is None:                                                                                                                                  
    if 'lr' in stats:                                                                                                                                      
        learning_rate = stats['lr'][-1]                                                                                                                    
    else:                                                                                                                                                  
        learning_rate = 0.001                                                                                                                              
                                                                                                                                                           
stats['lr'] = [learning_rate]                                                                                                                              
optimizer = AdamW(model.parameters(), lr=learning_rate)   # BOTH NEW                                                                                                 
scheduler = ReduceLROnPlateau(optimizer)                                                                                                                   
                                                                                                                                                    
for e in range(model_i+1, model_i + number_of_epochs+1): 
  trainingloss = [] 
  validationloss = []
  model.train()
  for x, y in train_dataloader:
    optimizer.zero_grad()
    x, y = x.to(device), y.to(device)

    pred = model(x)
    currloss = loss(pred, y)# + l1_param * model.linreg.weight.abs().sum()

    currloss.backward()

    optimizer.step()

    trainingloss.append(currloss.cpu().detach().numpy())
  #scheduler.step()

  with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      validationloss.append(loss(pred, y).cpu().detach().numpy())

    #num_zero = [torch.sum(l.weight == 0).cpu() for l in model.linreg]

    stats['train_average_loss'].append(np.mean(trainingloss))
    stats['validation_average_loss'].append(np.mean(validationloss))

    #stats['sparsity'].append((np.min(num_zero), np.max(num_zero)))

    print(f"Epoch {e+1}: Average loss (training) = {stats['train_average_loss'][-1]}")
    print(f"Epoch {e+1}: Average loss (validation) = {stats['validation_average_loss'][-1]}")

    #print(f"Epoch {e+1}: Min. number of zero = {np.min(num_zero)}, Max. number of zero = {np.max(num_zero)}")

    model.save_model(os.path.join(model_output, f"model.{e}"))

    stats['lr'].append(scheduler.optimizer.param_groups[0]['lr'])

    save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))

  scheduler.step(stats['validation_average_loss'][-1]) # Plateau scheduler 

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
