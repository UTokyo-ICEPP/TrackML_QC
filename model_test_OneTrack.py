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
import matplotlib.pyplot as plt
import networkx as nx

torch.manual_seed(0)

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


print('cuda_device : ', cuda_device)

from modules.OneTrackDataloader import  OneTrackDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model
#from modules.dynamic_graph_SC_en import Dynamic_Graph_Model

path = '/data/wachan/ds100/OneTrackSamples/evt1000/'

data_set_train = OneTrackDataset(path,num_start=1, num_end=10)
data_set_valid = OneTrackDataset(path,num_start=50, num_end=60)

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#model = Dynamic_Graph_Model(feature_dims_x = [4,5,3], feature_dims_en = [4,5,4])
model = Dynamic_Graph_Model(feature_dims_x = [4,6,8,10,3], feature_dims_en = [4,6,8,10,4])


model.to(cuda_device)
print( 'Model Cuda : ', next(model.parameters()).is_cuda )

data = next(iter(train_loader))
#print ('MAIN_DATA',data)

g, y, s = data

y_tar = model(g[0].to(cuda_device))

print ('nd : ',y)
print ('sc: ',s)
print ('y_tar [0]: ',y_tar[0])
print ('y_tar [1]: ',y_tar[1])
print('nd shape : ', y.shape)
print('sc shape:',s.shape)
print('y_tar [0] shape : ', y_tar[0].shape)
print('y_tar [1] shape : ', y_tar[1].shape)
print('s Device: ', s.device)
print('y_tar [1] Device : ', y_tar[1].device)
