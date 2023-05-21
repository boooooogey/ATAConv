#==== Libraries
#{{{
import sys
sys.path.append("/net/talisker/home/benos/mae117/Documents/research/chikina/ATAConv")

import torch, numpy as np
import os, pickle, argparse
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
import scipy.stats
import scipy.ndimage
import sklearn.metrics
import pyfaidx
import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from adam_penalty import AdamL1, AdamMCP
from torch.optim import AdamW
from utils import SeqDataset
from importlib import import_module
from torch.fft import rfft

#}}}

#==== Variables, Constants & More
#{{{
PREFIX_DATA  = "/net/talisker/home/benos/mae117/Documents/research/chikina/ATAConv/data"
FOLD_NUM = 9
#}}}

#==== Functions
#{{{
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
                stats, save_all, early_stopping,
                es_threshold, model_output, use_ftas):
  
  for e in range(model_i+1, model_i + number_of_epochs+1):
    trainingloss = [] 
    validationloss = []
    model.train()
    for x, y in train_dataloader:
      optimizer.zero_grad()
      x, y = x.to(device), y.to(device)

      if use_ftas:
        x.requires_grad = True  # Set gradient required
        pred = model(x)
        input_grads = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), retain_graph=True, create_graph=True)[0]
        input_grads = input_grads * x
        x.requires_grad = False
        currloss =  loss(pred, y) + fourier_loss(input_grads)
      
      else:
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

def place_tensor(tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = place_tensor(torch.tensor(kernel))

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(
        input_tensor, kernel, padding=sigma
    )

    return torch.squeeze(smoothed, dim=1)

def fourier_att_prior_loss(
    self, status, input_grads, freq_limit, limit_softness,
    att_prior_grad_smooth_sigma):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `status`: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0
            otherwise
        `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = torch.sum(torch.abs(input_grads), dim=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )

    # Only do the positives
    pos_grads = grads_smooth[status == 1]

    # Loss for positives
    if pos_grads.nelement():
        pos_fft = torch.rfft(pos_grads, 1)
        pos_mags = torch.norm(pos_fft, dim=2)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        weights = place_tensor(torch.ones_like(pos_mags))
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = place_tensor(
                torch.arange(1, pos_mags.size(1) - freq_limit + 1)
            ).float()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return place_tensor(torch.zeros(1))
#}}}

#==== Model Training
#{{{

params = dict()

use_ftas = False
meme_file = f"{PREFIX_DATA}/memes/cisBP_mouse.meme"   
signal_file = f"{PREFIX_DATA}/lineages/lineageImmgenDataCenterNK.txt" 
sequence_file = f"{PREFIX_DATA}/sequences/sequences.list"         
model_output = f"/net/talisker/home/benos/mae117/Documents/research/chikina/ATAConv/data/model_outputs/centernk_fourier-{use_ftas}/10foldcv/fold_{FOLD_NUM}"
split_folder = f"/net/talisker/home/benos/mae117/Documents/research/chikina/ATAConv/data/splits_all/10foldcv/fold_{FOLD_NUM}"
architecture_name = "model_pos_calib_sigmoid"
window_size = 300
params['number_of_epochs'] = 100
batch_size = 254
num_of_workers = 8
penalty_param = [0]
penalty_param_range = None
log_penalty_param_range = False
mcp_beta = 3
penalty_type = "l1"
model_name = None
learning_rate = None
step_learning_rate = None
unfreeze_conv = False
class_name = "TISFM"
params['save_all'] = False
params['early_stopping'] = True
params['es_threshold'] = 1e-7
penalize_layers = ['linreg.weight']
freeze_all_except_final = False
unfreeze_all = False

if "CenterNK" in signal_file:
  response_dim = 9
elif "diff" in signal_file:
  response_dim = 82
elif "full" in signal_file:
  response_dim = 90
else:
  response_dim = 8

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

params['train_dataloader'] = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_of_workers)
number_of_step = len(train_dataset)

params['test_dataloader'] = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

params['validation_dataloader'] = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

params['loss'] = torch.nn.MSELoss(reduction="mean")

stat_exists = os.path.exists(os.path.join(model_output, "stats.pkl"))

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
    params['scheduler'] = ReduceLROnPlateau(optimizer)
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
  params['use_ftas'] = use_ftas

  train_model(**params)
#}}}