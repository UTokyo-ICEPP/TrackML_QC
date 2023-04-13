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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


print('cuda_device : ', cuda_device)

from modules.TrackMLDataloader import  TrackMLDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model

path = '/data/wachan/ds100/'


data_set_train = TrackMLDataset(path, num_start=1000, num_end=1010)
data_set_valid = TrackMLDataset(path, num_start=1050, num_end=1054)

print(data_set_train)

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

model = Dynamic_Graph_Model(feature_dims_x = [4,3], feature_dims_en = [4, 3])

model.to(cuda_device)

print( 'Model Cuda : ', next(model.parameters()).is_cuda )

print ('TRAIN ',data_set_train[0])

data = next(iter(train_loader))#data_set_train[3]

#rint(data)

gr_list, y = data

print('N graphs : ', len(gr_list))

# x_new, e_new = gr.ndata['x'][0:1000], gr.ndata['en'][0:1000]

# del gr

# gr_n = dgl.knn_graph(x_new , 5)

# gr_n.ndata['x'] = x_new
# gr_n.ndata['en'] = e_new

y_tar = torch.cat([ model(ig.to(cuda_device)) for ig in gr_list]).mean().reshape(1,)#This is untrained?
# y = model(gr_n.to(cuda_device))
#y_tar: The number of doublet (in log) -> 'nd' in DataLoader?
print ('y : ',y)
print ('y_tar : ',y_tar)
print(y, y_tar)
print('y shape : ', y.shape)
print('y_tar shape : ', y_tar.shape)
