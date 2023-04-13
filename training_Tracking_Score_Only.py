import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import math

import uproot#3 as uproot
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

path = '/data/wachan/ds100/OneTrackSamples/evt1000_Nearby_Hit_v4_Random/'

#50%
data_set_train = OneTrackDataset(path,num_start=1, num_end=179701)
data_set_valid = OneTrackDataset(path,num_start=179702, num_end=359402)

#100%
#data_set_train = OneTrackDataset(path,num_start=1, num_end=359401)
#data_set_valid = OneTrackDataset(path,num_start=359402, num_end=718802)

#20%
#data_set_train = OneTrackDataset(path,num_start=1, num_end=71881)
#data_set_valid  = OneTrackDataset(path,num_start=71882, num_end=143762)

#Test untrain
#data_set_train = OneTrackDataset(path,num_start=1, num_end=1001)
#data_set_valid  = OneTrackDataset(path,num_start=1002, num_end=2002)

train_loader = DataLoader(data_set_train, batch_size=128, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=128, shuffle=False,collate_fn=collate_graphs, num_workers=0)

model = Dynamic_Graph_Model(feature_dims_x = [4,32,16,3], feature_dims_en = [4,32,16,4])
model_name = 'model_DynamicGraphTrack_Score_NH_v4_50p_batch_size_128.pt'

model.to(cuda_device)

print( 'Model Cuda : ', next(model.parameters()).is_cuda )

opt = optim.AdamW(model.parameters(), lr=0.00023489)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []

train_loss_s, valid_loss_s = [], []

# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
loss_fn_score = nn.CrossEntropyLoss()

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    #scheduler.step()
    model.train() ## --- set the model to train mode -- ##
    with tqdm(train_loader, ascii=True) as tq:
        for graph, label, score in tq:

            score = score.to(cuda_device)
            opt.zero_grad()

            pred_score = model(graph[0].to(cuda_device))[1]

            loss_score = loss_fn_score(pred_score, score)

            loss_score.backward()

            opt.step()

            train_loss += loss_score.item()

            del graph; del score; del pred_score;
            torch.cuda.empty_cache()



    #####################
    # validate the model #
    ######################
    model.eval()
    with tqdm(valid_loader, ascii=True) as tq:
        for graph, label, score in tq:

            score = score.to(cuda_device)

            pred_score = model(graph[0].to(cuda_device))[1]

            loss_score = loss_fn_score(pred_score, score)

            valid_loss += loss_score.item()

            del graph; del score; del pred_score;
            torch.cuda.empty_cache()

    # calculate average losses
    #Need to add here the score loss as well
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    train_loss_v.append(train_loss)
    valid_loss_v.append(valid_loss)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), model_name)
        valid_loss_min = valid_loss

# ---- end of script ------ #

hf = h5py.File('loss_epoch_file_Score_NH_v4_50p_batch_size_128.h5', 'w')
hf.create_dataset('train_loss', data=np.array(train_loss_v))
hf.create_dataset('valid_loss', data=np.array(valid_loss_v))
hf.close()
