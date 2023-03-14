from utils import SeqDataset
import torch, numpy as np
import os, pickle, argparse, pandas as pd
from importlib import import_module
from matplotlib import colors, pyplot as plt
import seaborn

####DELETE##################
from IPython import embed###
############################

def save_to_pickle(data, filepath):
  with open(filepath, "wb") as file:
    pickle.dump(data, file)

def read_from_pickle(filepath):
  with open(filepath, "rb") as file:
    data = pickle.load(file)
  return data

def plot_final_layer(model, names, top, path):
  ii, motif_names, final_layer = model.motif_ranks()
  top_motifs = ii[:, :top]
  top_ii = np.sort(np.unique(top_motifs.reshape(-1)))
  toplot = final_layer[:, top_ii]
  p = seaborn.clustermap(toplot, cmap="vlag", yticklabels=names, xticklabels=[i.split("+")[0] for i in motif_names[top_ii]], figsize=(16,9), col_cluster=True, row_cluster=True, dendrogram_ratio=(.1,.3), z_score=1)
  plt.setp(p.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  plt.tight_layout()
  plt.savefig(path)
  plt.close()

def evaluate_model(dataloader, loss, model, names, check_ranks):
  testloss = []
  
  with torch.no_grad():
    model.eval()
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      currloss = loss(pred, y).cpu().detach().numpy().mean(axis=0)
      testloss.append(currloss)

  testloss = np.vstack(testloss)
  testoverall = testloss.mean()
  testl = testloss.mean(axis=0)

  if check_ranks:
    ii, motif_names, _ = model.motif_ranks()

  b_i = np.where(names == "B")[0]
  innate_lym_i = np.where(names == "innate.lym")[0]
  out_stat = dict()

  out_stat["overall"] = [testoverall]
  print(f"Overall average MSE = {testoverall}")
  for i in range(len(testl)):
    out_stat[names[i]] = [testl[i]]
    print(f"Average MSE for {names[i]} = {testl[i]}")

  if check_ranks:
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

  return pd.DataFrame(out_stat)

def extract_and_write_final_layer(model, names, motif_names, model_name, file_name_template):
    with torch.no_grad():
        layer = model.linreg.weight.detach().cpu().numpy()
    for i, n in enumerate(names):
        file_name = f"{file_name_template}.{n}"
        out = pd.DataFrame({"motifs": motif_names, model_name: layer[i, :len(motif_names)]}).set_index("motifs")
        if os.path.exists(file_name):
            data = pd.read_csv(file_name, sep="\t", header=0, index_col=0)
            out = pd.concat([data, out], axis = 1)
        out.to_csv(file_name, sep="\t", header=True, index=True)

parser = argparse.ArgumentParser()
parser.add_argument("--meme-file", required=True, help="Path to the meme file that stores PWMs")
parser.add_argument("--atac-file", required=True, help="Path to the file that stores ATAC signal")
parser.add_argument("--sequences", required=True, help="Path to the file that stores sequences")
parser.add_argument("--model", required=True, help="Model file or directory")
parser.add_argument("--split-folder", required=True, help="Folder that stores train/val/test splits.")
parser.add_argument("--architecture", required=True, help="Architecture to be used.")
parser.add_argument("--window-size", default=300, type=int, help="Length of the sequence fragments")
parser.add_argument("--batch-size", default=254, type=int, help="Batch size")
parser.add_argument("--num-of-workers", default = 8, type=int, help="Number of workers for data loading")
parser.add_argument("--stat-out", default=None, help="The stats on the given model will be written to the file. Ignored if --model is a directory.")
parser.add_argument("--ai-atac", action="store_true", help="Do not check the final layer if the model is ai-atac.")
parser.add_argument("--class-name", default="TISFM", help="Model class name.")
parser.add_argument("--use-validation", action="store_true", help="Use validation split instead of test.")
parser.add_argument("--plot-path", action="store_true", help="If true, plots the results for the path algorithm.")
parser.add_argument("--plot-final-layer", action="store_true", help="If true, plots the final layer of the network.")
parser.add_argument("--plot-x-log", action="store_true", help="If true, log normalizes the x axis.")
parser.add_argument("--top-motifs", default=10, type=int, help="Top motifs for each cell types.")
parser.add_argument("--model-index", default=None, help="Model name to use when saving.")
parser.add_argument("--extract-final-layer", default=None, help="Extract and save the final layer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

meme_file = args.meme_file
signal_file = args.atac_file
sequence_file = args.sequences
model_name = args.model
split_folder = args.split_folder
architecture_name = args.architecture
window_size = args.window_size
batch_size = args.batch_size
num_of_workers = args.num_of_workers
file_stat = args.stat_out
isaiatac = args.ai_atac
class_name = args.class_name
usevalidation = args.use_validation
plotpath = args.plot_path
plotxlog = args.plot_x_log
plotfinallayer = args.plot_final_layer
topmotifs = args.top_motifs
model_index = args.model_index
extract_final_layer = args.extract_final_layer

dataset = SeqDataset(signal_file, sequence_file)

architecture = getattr(import_module(f"models.{architecture_name}"), class_name)
model = architecture(dataset.number_of_cell_types(), meme_file, window_size).to(device)

if usevalidation:
  file_name = "best_validation.tsv"
else:
  file_name = "best_test.tsv"

if usevalidation:
  indices = np.load(os.path.join(split_folder, "validation_split.npy"))
else:
  indices = np.load(os.path.join(split_folder, "test_split.npy"))

dataset_subset = torch.utils.data.Subset(dataset, indices)

dataloader = torch.utils.data.DataLoader(dataset_subset, batch_size = batch_size, shuffle = False, num_workers = num_of_workers)

loss = torch.nn.MSELoss(reduction="none")

names = dataset.cell_types()

if plotpath:
  penalty_param_list = read_from_pickle(os.path.join(model_name, "penalty_param_list.pkl"))
  path_length = len(penalty_param_list)
  colors = colors.TABLEAU_COLORS
  annotate_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
  other_color = 'tab:gray'
  res = []
  final_layers = []
  annotate_i = 6
  for i in range(path_length):
    model.load_model(os.path.join(model_name, f"path_{i}", "model.best"))
    if i == 0:
      motif_names = model.meme_file.motif_names() #motif_ranks() 
      n = len(motif_names)
    final_layers.append(model.linreg.weight.cpu().detach().numpy()[:,:n])
    res.append(evaluate_model(dataloader, loss, model, names, False))
    res[-1].index = [f"path_{i}"]
  res = pd.concat(res)
  to_plot_res = res.melt()
  res["lambda"] = penalty_param_list
  res.to_csv(os.path.join(model_name, "path_eval.tsv"), sep = "\t", header=True, index=True)
  to_plot_res.columns = ["Cell types", "MSE"]
  to_plot_res["lambda"] = np.tile(penalty_param_list, len(names)+1)

  fig,ax = plt.subplots(figsize=(16,9))
  lp = seaborn.lineplot(data=to_plot_res, x="lambda", y="MSE", hue="Cell types", ax=ax)
  if plotxlog:
    lp.set(xscale='log')
  fig.tight_layout()
  plt.savefig(os.path.join(model_name, "path_mse.png"))
  plt.close()

  for cell_name_i in range(len(names)):
    best_ii = np.argmin(res.iloc[:,cell_name_i].to_numpy())
    final_layer = np.vstack([i[cell_name_i] for i in final_layers])
    annotate = np.argsort(-np.abs(final_layer[best_ii]))[:9]
    others = np.argsort(-np.abs(final_layer[best_ii]))[9:]
    fig,ax = plt.subplots(figsize=(16,9))
    if plotxlog:
      plt.xscale("log") 
    for n, i in enumerate(others):
      ax.plot(penalty_param_list, final_layer[:,i], color=colors[other_color])
    for n, i in enumerate(annotate):
      ax.plot(penalty_param_list, final_layer[:,i], marker="$%s$" % motif_names[i].split('+')[0], color=colors[annotate_colors[n]], markersize=18)
    ax.axvline(x=penalty_param_list[best_ii], color='black', linestyle='--')
    ax.set_title(names[cell_name_i])
    fig.tight_layout()
    plt.savefig(os.path.join(model_name, f"path_coef_{names[cell_name_i]}.png"))
    plt.close()

else:
  if os.path.isdir(model_name):
    if os.path.exists(os.path.join(model_name, "model.best")):
      if file_stat is None:
          file_stat = os.path.join(model_name, file_name)
      model_name = os.path.join(model_name, f"model.best")
    else:
      stats = read_from_pickle(os.path.join(model_name, "stats.pkl"))
      ii = np.argmin(stats["validation_average_loss"])
      print(f"Best model validation MSE: {stats['validation_average_loss'][ii]}, Epoch: {stats['epoch'][ii]}.")
      if file_stat is None:
          file_stat = os.path.join(model_name, file_name)
      model_name = os.path.join(model_name, f"model.{stats['epoch'][ii]}")

  model.load_model(model_name)

  out = evaluate_model(dataloader, loss, model, names, not isaiatac)

  #keeping the path, could be useful later
  if model_index is not None:
    out.index = [model_index]
  else:
    out.index = [model_name]
  if os.path.exists(file_stat):
    stat = pd.read_csv(file_stat, sep="\t", header=0, index_col=0)
    pd.concat([stat, out]).to_csv(file_stat, sep="\t", header=True, index=True)
  else:
    out.to_csv(file_stat, sep="\t")
  if plotfinallayer:
    print(os.path.join(os.path.dirname(model_name), "final_layer.png"))
    plot_final_layer(model, names, topmotifs, os.path.join(os.path.dirname(model_name), "final_layer.png"))

if extract_final_layer is not None:
  print(f"Extracting final layer to {extract_final_layer}")
  motif_names = np.array(model.meme_file.motif_names())
  extract_and_write_final_layer(model, names, motif_names, model_index, extract_final_layer)
