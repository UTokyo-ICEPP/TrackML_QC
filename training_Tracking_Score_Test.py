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

import csv


torch.manual_seed(0)

import os, sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="choose the model type", type=str)
args = parser.parse_args()

model_name = args.model_name


os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_LAUNCH_BLOCKING"]='1'

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

print('cuda_device : ', cuda_device)

from modules.OneTrackDataloader import  OneTrackDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model

path = '/data/wachan/ds100/OneTrackSamples/evt1000/'


#data_set_train = OneTrackDataset(path,num_start=1, num_end=536)
#data_set_valid = OneTrackDataset(path,num_start=537, num_end=1072)

data_set_train = OneTrackDataset(path,num_start=1, num_end=2)
data_set_valid = OneTrackDataset(path,num_start=11, num_end=20)

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#model = Dynamic_Graph_Model(feature_dims_x = [4,6,8,10,12,3], feature_dims_en = [4,6,8,10,12,4])
model = Dynamic_Graph_Model(feature_dims_x = [4,6,8,10,3], feature_dims_en = [4,6,8,10,4])
#model = Dynamic_Graph_Model(feature_dims_x = [4,6,18,3], feature_dims_en = [4,6,18,4])
model_name = 'model_DynamicGraphTrack_Score_Test_NewOutput_2.pt'

model.to(cuda_device)

print( 'Model Cuda : ', next(model.parameters()).is_cuda )

#opt = optim.AdamW(model.parameters(), lr=1e-4)
opt = optim.AdamW(model.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []

train_loss_s, valid_loss_s = [], []


# number of epochs to train the model
#n_epochs = 10
n_epochs = 500

valid_loss_min = np.Inf # track change in validation loss

loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
loss_fn_score = nn.CrossEntropyLoss()

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    train_loss_sc = 0.0

    ###################
    # train the model #
    ###################
    #scheduler.step()
    model.train() ## --- set the model to train mode -- ##
    with tqdm(train_loader, ascii=True) as tq:
        for graph, label, score in tq:

            #label = label.to(cuda_device).float()
            score = score.to(cuda_device)
            opt.zero_grad()

            #pred_label = [ model(ig.to(cuda_device)) for ig in gr_list][0][0]
            pred_score = model(graph[0].to(cuda_device))[1]
            #print (pred_score)

            #loss = loss_fn(pred_label, label)
            loss_score = loss_fn_score(pred_score, score)
            total_loss = loss_score

            total_loss.backward()

            opt.step()

            train_loss += total_loss.item()

            del graph; del score; del pred_score;
            torch.cuda.empty_cache()

    # calculate average losses
    #Need to add here the score loss as well
    train_loss = train_loss/len(train_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, train_loss))#, valid_loss))

    train_loss_v.append(train_loss)
    #train_loss_s.append(train_loss_sc)

hf = h5py.File('loss_epoch_file_Train_Only_3LayerTest_NewOutput_2.h5', 'w')
hf.create_dataset('train_loss', data=np.array(train_loss_v))
hf.close()

# ---- end of script ------ #
