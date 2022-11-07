from utils import SeqDataset
import torch, numpy as np
import os, pickle, argparse, pandas as pd
from importlib import import_module

# added for benchmarking
from models.aitac import ConvNet

#==== Constants

PREFIX_DATA  = "/net/talisker/home/benos/mae117/collaborations/atac_convolutions/data"
PREFIX_AITAC = "/net/talisker/home/benos/mae117/collaborations/atac_convolutions/AI-TAC"
meme         = f"{PREFIX_DATA}/cisBP_mouse.meme"                
bed          = f"{PREFIX_DATA}/lineageImmgenDataZ.txt"         
seq          = f"{PREFIX_DATA}/sequences.list"                  
out          = f"{PREFIX_DATA}/model_outputs/aitac/10foldcv/fold_9" 
splits       = f"{PREFIX_DATA}/10foldcv/fold_9"
model_v      = f"model.0"
file         = f"aitac"                  

#==== Functions

def save_to_pickle(data, filepath):
  with open(filepath, "wb") as file:
    pickle.dump(data, file)

def read_from_pickle(filepath):
  with open(filepath, "rb") as file:
    data = pickle.load(file)
  return data

if __name__=="__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--meme_file",    default=meme, help="Path to the meme file that stores PWMs")
  parser.add_argument("--atac_file",    default=bed, help="Path to the file that stores ATAC signal")
  parser.add_argument("--sequences",    default=seq, help="Path to the file that stores sequences")
  parser.add_argument("--model_output", default=out, help="Directory to store model parameters")
  parser.add_argument("--split_folder", default=splits, help="Folder that stores train/val/test splits.")
  parser.add_argument("--architecture", default=file, help="Architecture to be used.")
  parser.add_argument("--window_size", default=300, type=int, help="Length of the sequence fragments")
  parser.add_argument("--batch_size", default=254, type=int, help="Batch size")
  parser.add_argument("--num_of_workers", default=8, type=int, help="Number of workers for data loading")
  parser.add_argument("--response_num", default=8, type=int, help="number of signals to predict")
  parser.add_argument("--class_name", default="ConvNet", help="Model class name.")

  # add
  parser.add_argument("--model", default=out, help="Model file or directory")
  parser.add_argument("--stat_out", help="The stats on the given model will be written to the file. Ignored if --model is a directory.")
  parser.add_argument("--ai-atac", default=True, help="Do not check the final layer if the model is ai-atac.")

  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  meme_file = args.meme_file
  signal_file = args.atac_file
  sequence_file = args.sequences
  model_name = args.model
  split_folder = args.split_folder
  architecture_name = args.architecture
  window_size = args.window_size
  batch_size = args.batch_size
  num_of_workers = args.num_of_workers
  response_dim = args.response_num
  file_stat = args.stat_out
  isaiatac = args.ai_atac
  class_name = args.class_name

  architecture = getattr(import_module(f"models.{architecture_name}"), class_name)

  # input depends on architecture
  if architecture_name == "aitac":
    model = architecture(num_classes=response_dim, num_filters=300).to(device)
  else:
    model = architecture(response_dim, meme_file, window_size).to(device)

  if os.path.isdir(model_name):
    stats = read_from_pickle(os.path.join(model_name, "stats.pkl"))
    ii = np.argmin(stats["validation_average_loss"])
    print(f"Best model validation MSE: {stats['validation_average_loss'][ii]}, Epoch: {stats['epoch'][ii]}.")
    file_stat = os.path.join(model_name, "best.tsv")
    model_name = os.path.join(model_name, f"model.{stats['epoch'][ii]}")

  model.load_model(model_name)

  dataset = SeqDataset(signal_file, sequence_file)

  test_indices = np.load(os.path.join(split_folder, "test_split.npy"))
  test_dataset = torch.utils.data.Subset(dataset, test_indices)

  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

  loss = torch.nn.MSELoss(reduction="none")

  testloss = []

  names = dataset.cell_types()
  ii, motif_names = model.motif_ranks()
  b_i = np.where(names == "B")[0]
  innate_lym_i = np.where(names == "innate.lym")[0]
  out_stat = dict()

  with torch.no_grad():
    model.eval()
    for x, y in test_dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      currloss = loss(pred, y).cpu().detach().numpy().mean(axis=0)
      testloss.append(currloss)

  testl = np.vstack(testloss).mean(axis=0)
  for i in range(len(testl)):
    out_stat[names[i]] = [testl[i]]
    print(f"Average MSE for {names[i]} = {testl[i]}")

  if not isaiatac:
    pax5_ii = np.where(motif_names[ii[b_i]] == "Pax5+M1848_1.02+I")[1]
    out_stat["Pax5"] = pax5_ii

    irf4_ii = np.where(motif_names[ii[b_i]] == "Irf4+M1264_1.02+D")[1]
    out_stat["Irf4"] = irf4_ii

    ebf1_ii = np.where(motif_names[ii[b_i]] == "Ebf1+M3690_1.02+D")[1]
    out_stat["Ebf1"] = ebf1_ii

    rorc_ii = np.where(motif_names[ii[innate_lym_i]] == "Rorc+M6455_1.02+I")[1]
    out_stat["Rorc"] = rorc_ii

    print(f"Pax5's rank for B: {pax5_ii[0]}")
    print(f"Irf4's rank for B: {irf4_ii[0]}")
    print(f"Ebf1's rank for B: {ebf1_ii[0]}")
    print(f"Rorc's rank for innate.lym: {rorc_ii[0]}")

  out = pd.DataFrame(out_stat)
  #keeping the path, could be useful later
  out.index = [model_name]
  out.to_csv(file_stat, sep="\t")
