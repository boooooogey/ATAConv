#!/bin/bash


#==== Define Constants
PREFIX_DATA="/net/talisker/home/benos/mae117/collaborations/tiSFM/data"
PREFIX_AITAC= ="/net/talisker/home/benos/mae117/collaborations/tiSFM/AI-TAC"
MEME=f"{PREFIX_DATA}/cisBP_mouse.meme"                

FOLD_NUM=9
BED=f"{PREFIX_DATA}/lineageImmgenDataCenterNK.txt"         

seq    = f"{PREFIX_DATA}/sequences.list"                  
out    = f"{PREFIX_DATA}/model_outputs/aitac_centernk/10foldcv/fold_{FOLD_NUM}"
splits = f"{PREFIX_DATA}/10foldcv/fold_{FOLD_NUM}"
file   = f"aitac"                  

./train_model.py --meme_file ""
