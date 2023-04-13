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
#os.environ["CUDA_LAUNCH_BLOCKING"]='1'

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


print('cuda_device : ', cuda_device)

from modules.TrackMLDataloader import  TrackMLDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model
#from modules.lorentz_graph import LorentzGroup_Model
#from modules.attention_graph import Graph_Attention_Model

path = '/data/wachan/ds100/'


data_set_train = TrackMLDataset(path, num_start=1000, num_end=1050)
data_set_valid = TrackMLDataset(path, num_start=1050, num_end=1060)

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#print(data_set_train[0])

if(model_name == 'edgeconv') :
    model = Dynamic_Graph_Model(feature_dims_x = [4,3,4], feature_dims_en = [4, 2, 3]) # feature_dims_x = [3, 5, 4, 2], feature_dims_en = [4, 5, 6, 8]
    model_name = 'model_DynamicGraphTrack.pt'
# elif(model_name == 'lorentz') :
#     model = LorentzGroup_Model(init_dim=[1,2], feature_dims_x=[5,6,7,4], feature_dims_h=[6,7,8,5], feature_dims_m=[3,4,5,4],\
#                                 device=cuda_device)
#     model_name = 'model_LorentzJet.pt'
# else :
#     model = Graph_Attention_Model(num_heads = 5, feature_dims = [10, 15, 12, 8], input_names=cluster_var)
#     model_name = 'model_AttentionGraphJet.pt'

model.to(cuda_device)

print( 'Model Cuda : ', next(model.parameters()).is_cuda )

opt = optim.AdamW(model.parameters(), lr=1e-2)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []


# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

loss_fn = nn.MSELoss()

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
        for gr_list, label in tq:

            label = label.to(cuda_device)

            opt.zero_grad()

            pred_label = torch.cat([ model(ig.to(cuda_device)) for ig in gr_list]).mean().reshape(1,)

            loss = loss_fn(pred_label, label)

            loss.backward()
            #loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            opt.step()
            #scheduler.step()

            # update training loss
            train_loss += loss.item()

            del gr_list; del label; del pred_label;
            torch.cuda.empty_cache()


    #####################
    # validate the model #
    ######################
    model.eval()
    with tqdm(valid_loader, ascii=True) as tq:
        for gr_list, label in tq:

            label = label.to(cuda_device)

            pred_label = torch.cat([ model(ig.to(cuda_device)) for ig in gr_list]).mean().reshape(1,)

            loss = loss_fn(pred_label, label)

            valid_loss += loss.item()

            del gr_list; del label; del pred_label;
            torch.cuda.empty_cache()


    # calculate average losses
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

hf = h5py.File('loss_epoch_file.h5', 'w')
hf.create_dataset('train_loss', data=np.array(train_loss_v))
hf.create_dataset('valid_loss', data=np.array(valid_loss_v))
hf.close()




print('Hello')
