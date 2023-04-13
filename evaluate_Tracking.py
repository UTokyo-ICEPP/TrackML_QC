import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import math

import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.manual_seed(0)

import os, sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="choose the model type", type=str)
args = parser.parse_args()

model_name = args.model_name


os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

print('cuda_device : ', cuda_device)

from modules.TrackMLDataloader import  TrackMLDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model

path = '/data/wachan/ds100/'

data_set_test = TrackMLDataset(path, num_start=1060, num_end=1099)

test_loader = DataLoader(data_set_test, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)


if(model_name == 'edgeconv') : 
    model = Dynamic_Graph_Model(feature_dims_x = [4,3,4], feature_dims_en = [4, 2, 3])
    model_name = 'model_DynamicGraphTrack.pt'
    out_name = 'PredictionFile_DynamicGraphTrack.h5'
# elif(model_name == 'lorentz') : 
#     model = LorentzGroup_Model(init_dim=[1,2], feature_dims_x=[5,6,7,4], feature_dims_h=[6,7,8,5], feature_dims_m=[3,4,5,4],\
#                                 device=cuda_device)
#     model_name = 'model_LorentzJet.pt'
#     out_name = 'PredictionFile_LorentzNetDiHiggs.h5'
# else : 
#     model = Graph_Attention_Model(num_heads = 5, feature_dims = [10, 15, 12, 8], input_names=cluster_var)
#     model_name = 'model_AttentionGraphJet.pt'
#     out_name = 'PredictionFile_AttentionGraphDiHiggst.h5'

model.to(cuda_device)

model.load_state_dict(torch.load(model_name))

param_numb = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters : ', param_numb)

model.eval()

truth_labels, predicted_labels = [], [] 

with tqdm(test_loader, ascii=True) as tq:
    for gr_list, label in tq:
            
	    #label = label.to(cuda_device)
	    
	    pred_label = torch.cat([ model(ig.to(cuda_device)) for ig in gr_list]).mean().reshape(1,)
	    
	    # print('label shape : ', label.shape)
	    # print('pred label shape : ', pred_label.shape)

	    truth_labels.append(label.cpu().detach().numpy())
	    predicted_labels.append(pred_label.cpu().detach().numpy())
	    
	    del gr_list; del label; del pred_label;

truth_labels, predicted_labels = np.concatenate(truth_labels), np.concatenate(predicted_labels)

hf = h5py.File(out_name, 'w')

hf.create_dataset('truth_labels', data=truth_labels, compression='lzf')
hf.create_dataset('predicted_labels', data=predicted_labels, compression='lzf')

hf.close()

