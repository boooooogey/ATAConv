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

def freeze_all_except_final_func(model):
    model.freeze(doall=True)
    model.unfreeze(layers=['linreg'])
    return model

def model_freeze_unfreeze_layers(model, unfreeze_conv, unfreeze_all, freeze_all_except_final):
    if unfreeze_conv:
        model.unfreeze()
    elif unfreeze_all:
        model.unfreeze(doall=True)
    elif freeze_all_except_final:
        model = freeze_all_except_final_func(model)
    else:
        pass
    return model

def train_model(model, model_i, number_of_epochs, 
                train_dataloader, validation_dataloader, test_dataloader, 
                loss, optimizer, scheduler, 
                stats, save_all, early_stopping, es_threshold, model_output):
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
        if np.argmin(stats['train_average_loss']) == (len(stats['train_average_loss'])-1):
          print("Training MSE improved. Saving the model.")
          model.save_model(os.path.join(model_output, "model.train.best"))

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

def return_optimizer(model, penalty_param, penalty_type, learning_rate, targets):
  if penalty_param != 0:
    if penalty_type == "l1":
      optimizer = AdamL1([{'params': [i[1] for i in filter(lambda x: not x[0] in targets, model.named_parameters())],
                           'penalty_hyper_param': 0},
                          {'params': [i[1] for i in filter(lambda x: x[0] in targets, model.named_parameters())],
                           'penalty_hyper_param': penalty_param}], lr=learning_rate)
    elif penalty_type == "mcp":
      optimizer = AdamMCP([{'params': [i[1] for i in filter(lambda x: not x[0] in targets, model.named_parameters())],
                            'penalty_hyper_param': 0},
                           {'params': [i[1] for i in filter(lambda x: x[0] in targets, model.named_parameters())],
                            'penalty_hyper_param': penalty_param,
                            'b': 3}], lr=learning_rate)
    else:
      raise ValueError(f"Invalid penalty type {penalty_type}.")
  else:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
  return optimizer

def empty_stats():
  stats = {
      'train_average_loss' : [],
      'validation_average_loss' : [],
      'epoch': [],
      'lr': [],
      'test_average_loss' : 0,
      'completed' : False
      }
  return stats

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
parser.add_argument("--penalty-param", nargs='+', default=[0], type=float, help="Hyperparameter for the regularization of the final layer")
parser.add_argument("--penalty-param-range", nargs=3, default=None, type=float, help="Linear range for the path algorithm as --penalty-param-range start stop number. The other options are ignored if this one is given.")
parser.add_argument("--log-penalty-param-range", action="store_true", help="Use log range instead of the linear range.")
parser.add_argument("--mcp-param", default=3, type=float, help="Second hyperparameter for the mcp regularization of the final layer (first is --penalty_param)")
parser.add_argument("--penalty-type", default="l1", help="l1/mcp regularization")
parser.add_argument("--model", default=None, help="Start from the given state.")
parser.add_argument("--lr", default=None, type=float, help="Learning rate")
parser.add_argument("--step-lr", default=None, type=float, help="For path algorithm.")
parser.add_argument("--unfreeze-conv", action='store_true', help="Whether to unfreeze the motifs or to keep them frozen (default: false)") 
parser.add_argument("--class-name", default="TISFM", help="Model class name.")
parser.add_argument("--save-all", action='store_true', help="Whether to save at the end of every epoch.") 
parser.add_argument("--early-stopping", action='store_true', help="Whether to stop training early if validation MSE does not improve.") 
parser.add_argument("--early-stopping-threshold", default=1e-7, type=float, help="If early stopping is set, the learning rate is checked to decide whether to stop training.") 
parser.add_argument("--penalize-layers", default=['linreg.weight'], type=str, help="on which layers, the penalty should be imposed.") 
parser.add_argument("--freeze-all-except-final-layer", action='store_true', help="If set, all the layers are frozen except for the final layer.") 
parser.add_argument("--unfreeze-all", action='store_true', help="Unfreeze all layers.") 

args = parser.parse_args()

params = dict()

meme_file = args.meme_file
signal_file = args.atac_file
sequence_file = args.sequences
model_output = args.model_output
split_folder = args.split_folder
architecture_name = args.architecture
window_size = args.window_size
params['number_of_epochs'] = args.number_of_epochs
batch_size = args.batch_size
num_of_workers = args.num_of_workers
penalty_param = args.penalty_param
penalty_param_range = args.penalty_param_range
log_penalty_param_range = args.log_penalty_param_range
mcp_beta = args.mcp_param
penalty_type = args.penalty_type
model_name = args.model
learning_rate = args.lr
step_learning_rate = args.step_lr
unfreeze_conv = args.unfreeze_conv
class_name = args.class_name
params['save_all'] = args.save_all
params['early_stopping'] = args.early_stopping
params['es_threshold'] = args.early_stopping_threshold
penalize_layers = args.penalize_layers
freeze_all_except_final = args.freeze_all_except_final_layer
unfreeze_all = args.unfreeze_all

if model_name is None or model_name.endswith("model.best"):
    model_i = -1
else:
    model_i = int(model_name.split(".")[-1])

os.path.exists(model_output) or os.makedirs(model_output)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device_name}")

dataset = SeqDataset(signal_file, sequence_file)

architecture = getattr(import_module(f"models.{architecture_name}"), class_name)
model = architecture(dataset.number_of_cell_types(), meme_file, window_size).to(device)

if model_name is None:
    model.init_weights()
else:
    model.load_model(model_name)

model = model_freeze_unfreeze_layers(model, unfreeze_conv, unfreeze_all, freeze_all_except_final)

train_indices = np.load(os.path.join(split_folder, "train_split.npy"))
validation_indices = np.load(os.path.join(split_folder, "validation_split.npy"))
test_indices = np.load(os.path.join(split_folder, "test_split.npy"))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

params['train_dataloader'] = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size = batch_size,
                                                         shuffle = True, pin_memory = True,
                                                         num_workers = num_of_workers,
                                                         drop_last = True)
number_of_step = len(train_dataset)

params['test_dataloader'] = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = False,
                                                        num_workers = num_of_workers,
                                                        drop_last = True)

params['validation_dataloader'] = torch.utils.data.DataLoader(validation_dataset,
                                                              batch_size = batch_size,
                                                              shuffle = False,
                                                              num_workers = num_of_workers,
                                                              drop_last = True)

params['loss'] = torch.nn.MSELoss(reduction="mean")

stat_exists = os.path.exists(os.path.join(model_output, "stats.pkl"))
if model_i != -1 and stat_exists:
    with open(os.path.join(model_output, "stats.pkl"), "rb") as file:
        stats = pickle.load(file)
    stats['train_average_loss'] = stats['train_average_loss'][:model_i+1]
    stats['validation_average_loss'] = stats['validation_average_loss'][:model_i+1]
    stats['lr'] = stats['lr'][:model_i+1]
    stats['epoch'] = stats['epoch'][:model_i+1]
elif  stat_exists and model_name is not None and model_name.endswith("model.best"):
    with open(os.path.join(model_output, "stats.pkl"), "rb") as file:
        stats = pickle.load(file)
    model_i =stats['epoch'][-1]
else:
    stats = empty_stats()

if learning_rate is None:
    if 'lr' in stats and len(stats['lr']) > 0:
        learning_rate = stats['lr'][-1]
    else:
        learning_rate = 0.001
        #stats

if step_learning_rate is None:
  step_learning_rate = learning_rate / 10

#if isinstance(penalty_param, list):
if len(penalty_param) != 1 or penalty_param_range is not None:
  print("Initializing path")
  if penalty_param_range is not None:
    start = penalty_param_range[0]
    end = penalty_param_range[1]
    num = penalty_param_range[2]
    if log_penalty_param_range:
      penalty_param_list = np.sort(np.unique(np.concatenate([[0],np.exp(np.linspace(start=np.log(start), stop=np.log(end), num=int(num)))])))
      #penalty_param_list = np.sort(np.unique(np.exp(np.linspace(start=np.log(start), stop=np.log(end), num=int(num)))))
    else:
      penalty_param_list = np.sort(np.unique(np.concatenate([[0],np.linspace(start=start, stop=end, num=int(num))])))
  else:
    penalty_param_list = np.sort(np.unique((np.array(penalty_param))))
  print(penalty_param_list)
  params['model_i'] = -1
  params['save_all'] = False

  save_to_pickle(penalty_param_list, os.path.join(model_output, "penalty_param_list.pkl"))
  continue_i = np.array([int(i.split("_")[1]) if i.startswith("path") else -1 for i in os.listdir(model_output)]).max()
  for i, penalty_param in enumerate(penalty_param_list):
    print(f"Step {i}: penalty parameter = {penalty_param:.5f}")
    model = model_freeze_unfreeze_layers(model, unfreeze_conv, unfreeze_all, freeze_all_except_final)
    params['model'] = model
    params['stats'] = empty_stats()
    path_dir = os.path.join(model_output, f"path_{i}")
    os.path.exists(path_dir) or os.mkdir(path_dir)
    params['model_output'] = path_dir
    if i == 0:
      optimizer = return_optimizer(model, penalty_param, penalty_type, learning_rate, penalize_layers)
    else:
      optimizer = return_optimizer(model, penalty_param, penalty_type, step_learning_rate, penalize_layers)
    params['scheduler'] = ReduceLROnPlateau(optimizer, patience=20)
    params['optimizer'] = optimizer
   
    if continue_i > i:
        print("Skipping")
        model.load_model(os.path.join(path_dir, "model.best"))
    else:
        train_model(**params)

    model.load_model(os.path.join(path_dir, "model.best"))
else:
  model = model_freeze_unfreeze_layers(model, unfreeze_conv, unfreeze_all, freeze_all_except_final)
  penalty_param = penalty_param[0]
  optimizer = return_optimizer(model, penalty_param, penalty_type, learning_rate, penalize_layers)

  params['scheduler'] = ReduceLROnPlateau(optimizer)
   
  params['stats'] = stats
  params['optimizer'] = optimizer
  params['model'] = model
  params['model_i'] = model_i
  params['model_output'] = model_output

  train_model(**params)
