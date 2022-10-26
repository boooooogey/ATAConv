from torch.optim.lr_scheduler import ReduceLROnPlateau
from adam_penalty import AdamL1, AdamMCP
from torch.optim import AdamW
from utils import SeqDataset
import torch, numpy as np
import os, pickle, argparse
from importlib import import_module

def save_to_pickle(data, filepath):
  with open(filepath, "wb") as file:
    pickle.dump(data, file)

parser = argparse.ArgumentParser()
parser.add_argument("--meme-file", required=True, help="Path to the meme file that stores PWMs")
parser.add_argument("--atac-file", required=True, help="Path to the file that stores ATAC signal")
parser.add_argument("--sequences", required=True, help="Path to the file that stores sequences")
parser.add_argument("--model-output", required=True, help="Directory to store model parameters")
parser.add_argument("--split-folder", required=True, help="Folder that stores train/val/test splits.")
parser.add_argument("--architecture", required=True, help="Architecture to be used.")
parser.add_argument("--window-size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--number-of-epochs", default=10, type=int, help="Number of epochs for training")
parser.add_argument("--batch-size", default=254, type=int, help="Batch size")
parser.add_argument("--num-of-workers", default=8, type=int, help="Number of workers for data loading")
parser.add_argument("--penalty-param", default=0, type=float, help="Hyperparameter for the regularization of the final layer")
parser.add_argument("--mcp-param", default=3, type=float, help="Second hyperparameter for the mcp regularization of the final layer (first is --penalty_param)")
parser.add_argument("--penalty-type", default="l1", help="l1/mcp regularization")
parser.add_argument("--model", default=None, help="Start from the given state.")
parser.add_argument("--lr", default=None, type=float, help="Learning rate")
parser.add_argument("--response-num", default=8, type=int, help="number of signals to predict")
parser.add_argument("--unfreeze-conv", action='store_true', help="Whether to unfreeze the motifs or to keep them frozen (default: false)") 
parser.add_argument("--class-name", default="TISFM", help="Model class name.")
parser.add_argument("--save-all", action='store_true', help="Whether to save at the end of every epoch.") 
parser.add_argument("--early-stopping", action='store_true', help="Whether to stop training early if validation MSE does not improve.") 
parser.add_argument("--early-stopping-threshold", default=1e-7, help="If early stopping is set, the learning rate is checked to decide whether to stop training.") 

args = parser.parse_args()

meme_file = args.meme_file
signal_file = args.atac_file
sequence_file = args.sequences
model_output = args.model_output
split_folder = args.split_folder
architecture_name = args.architecture
window_size = args.window_size
number_of_epochs = args.number_of_epochs
batch_size = args.batch_size
num_of_workers = args.num_of_workers
penalty_param = args.penalty_param
mcp_beta = args.mcp_param
penalty_type = args.penalty_type
model_name = args.model
learning_rate = args.lr
response_dim = args.response_num
unfreeze = args.unfreeze_conv
class_name = args.class_name
save_all = args.save_all
early_stopping = args.early_stopping
es_threshold = args.early_stopping_threshold

if model_name is None or model_name.endswith("model.best"):
    model_i = -1
else:
    model_i = int(model_name.split(".")[-1])

os.path.exists(model_output) or os.makedirs(model_output)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device_name}")

architecture = getattr(import_module(f"models.{architecture_name}"), class_name)
model = architecture(response_dim, meme_file, window_size).to(device)
if unfreeze:
  model.unfreeze()


if model_name is None:
    model.init_weights()
else:
    model.load_model(model_name)

dataset = SeqDataset(signal_file, sequence_file)

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

if model_i != -1 and os.path.exists(os.path.join(model_output, "stats.pkl")):
    with open(os.path.join(model_output, "stats.pkl"), "rb") as file:
        stats = pickle.load(file)
    stats['train_average_loss'] = stats['train_average_loss'][:model_i+1]
    stats['validation_average_loss'] = stats['validation_average_loss'][:model_i+1]
    stats['lr'] = stats['lr'][:model_i+1]
    stats['epoch'] = stats['epoch'][:model_i+1]
elif model_name.endswith("model.best"):
    with open(os.path.join(model_output, "stats.pkl"), "rb") as file:
        stats = pickle.load(file)
    model_i =stats['epoch'][-1]
else:
    stats = {
        'train_average_loss' : [],
        'validation_average_loss' : [],
        'epoch': [],
        'test_average_loss' : 0,
        'completed' : False
        }

if learning_rate is None:
    if 'lr' in stats:
        learning_rate = stats['lr'][-1]
    else:
        learning_rate = 0.001

if penalty_param != 0:
  if penalty_type == "l1":
    optimizer = AdamL1([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
                         'penalty_hyper_param': 0},
                        {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
                         'penalty_hyper_param': penalty_param}], lr=learning_rate)
  elif penalty_type == "mcp":
    optimizer = AdamMCP([{'params': [i[1] for i in filter(lambda x: not x[0].startswith("linreg"), model.named_parameters())],
                          'penalty_hyper_param': 0},
                         {'params': [i[1] for i in filter(lambda x: x[0].startswith("linreg"), model.named_parameters())],
                          'penalty_hyper_param': penalty_param,
                          'b': 3}], lr=learning_rate)
  else:
    raise ValueError(f"Invalid penalty type {penalty_type}.")
else:
  optimizer = AdamW(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer)
stats['lr'] = []

for e in range(model_i+1, model_i + number_of_epochs+1):
  trainingloss = [] 
  validationloss = []
  model.train()
  for x, y in train_dataloader:
    optimizer.zero_grad()
    x, y = x.to(device), y.to(device)

    pred = model(x)
    currloss = loss(pred, y)

    currloss.backward()

    optimizer.step()

    trainingloss.append(currloss.cpu().detach().numpy())

  with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      validationloss.append(loss(pred, y).cpu().detach().numpy())

    stats['train_average_loss'].append(np.mean(trainingloss))
    stats['validation_average_loss'].append(np.mean(validationloss))
    stats['lr'].append(scheduler.optimizer.param_groups[0]['lr'])
    stats['epoch'].append(e)
    print(f"Epoch {e}: Average loss (training) = {stats['train_average_loss'][-1]:.5f}\tAverage loss (validation) = {stats['validation_average_loss'][-1]:.5f}\nLearning rate = {scheduler.optimizer.param_groups[0]['lr']}")
    if save_all:
      model.save_model(os.path.join(model_output, f"model.{e}"))
    else:
      if np.argmin(stats['validation_average_loss']) == (len(stats['validation_average_loss'])-1):
        print("Validation MSE improved. Saving the model.")
        model.save_model(os.path.join(model_output, "model.best"))

    save_to_pickle(stats, os.path.join(model_output, f"stats.pkl"))

  if early_stopping and scheduler.optimizer.param_groups[0]['lr'] <= es_threshold:
    print("Early stopping!")
    break

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
