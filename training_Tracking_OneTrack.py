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

data_set_train = OneTrackDataset(path,num_start=1, num_end=50)
data_set_valid = OneTrackDataset(path,num_start=51, num_end=100)

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#model = Dynamic_Graph_Model(feature_dims_x = [4,3,4], feature_dims_en = [4, 3, 4])
model = Dynamic_Graph_Model(feature_dims_x = [4,3], feature_dims_en = [4, 4])
model_name = 'model_DynamicGraphTrack.pt'

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

            label = label.to(cuda_device).float()

            opt.zero_grad()

            pred_label = torch.cat([ model(ig.to(cuda_device)) for ig in gr_list]).mean().reshape(1,)

            loss = loss_fn(pred_label, label)

            #loss.backward()
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            opt.step()
            #scheduler.step()

            # update training loss
            train_loss += loss.item()

            Num_Doublet = label.tolist()
            Pred_Num_Doublet = pred_label.tolist()

            #print(Num_Doublet[0],Pred_Num_Doublet[0])

            label_list = [Num_Doublet[0],Pred_Num_Doublet[0]]

            #print (label_list)
            if epoch == 1:
                with open('Test_Num_Doublet_Train_2_1.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)
            elif epoch == 2:
                with open('Test_Num_Doublet_Train_2_2.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)
            elif epoch == n_epochs:
                with open('Test_Num_Doublet_Train_2_Last.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)

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

            Num_Doublet = label.tolist()
            Pred_Num_Doublet = pred_label.tolist()

            #print(Num_Doublet[0],Pred_Num_Doublet[0])

            label_list = [Num_Doublet[0],Pred_Num_Doublet[0]]

            #print (label_list)
            if epoch == 1:
                with open('Test_Num_Doublet_Vaild_2_1.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)
            elif epoch == 2:
                with open('Test_Num_Doublet_Vaild_2_2.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)
            elif epoch == n_epochs:
                with open('Test_Num_Doublet_Vaild_2_Last.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_list)

            #sys.exit()

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

"""
hf = h5py.File('loss_epoch_file.h5', 'w')
hf.create_dataset('train_loss', data=np.array(train_loss_v))
hf.create_dataset('valid_loss', data=np.array(valid_loss_v))
hf.close()
"""

#print (train_loss_v)
#print (valid_loss_v)

Output_list = list(zip(*[train_loss_v,valid_loss_v]))

#print (Output_list)
for i in range(len(Output_list)):
    with open('Test_Loss_10.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(Output_list[i])

#Why not making graph from lists directly?
