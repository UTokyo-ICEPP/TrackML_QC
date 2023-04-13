import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch

import dgl
from dgl import backend as F
#from dgl import RemoveSelfLoop

import math

from torch.utils.data import Dataset, DataLoader, Sampler

class OneTrackDataset(Dataset):
    def __init__(self, filepath, num_start, num_end):

        self.path = filepath
        self.num_start = num_start
        self.num_end = num_end

        self.n_eff = self.num_end - self.num_start

    def __len__(self):

        return self.n_eff

    def __getitem__(self, event_idx):

        # ------- building the input truth particle graph ---------- #

        event_idx = self.num_start + event_idx
        #event_idx = 4
        #print('Eve_ID',event_idx)

        hits= pd.read_csv(self.path + 'OneTrackSample00%s-hits.csv'%event_idx)
        doublets = pd.read_csv(self.path + 'OneTrackSample00%s-doublets.csv'%event_idx)
        #print (hits)
        #print (doublets)

        hit_x = torch.tensor(hits.to_numpy()[:,1])
        hit_y = torch.tensor(hits.to_numpy()[:,2])
        hit_z = torch.tensor(hits.to_numpy()[:,3])

        #Calculate r
        hits_r = []

        for i in range(len(hit_x)):
            r = math.sqrt((hit_x[i])*(hit_x[i])+(hit_y[i])*(hit_y[i]))
            hits_r.append(r)

        #print(hits_r)

        hit_r = torch.FloatTensor(hits_r)
        #print(hit_r)

        #node features: Positions
        pos = torch.stack([hit_x, hit_y, hit_z], dim=-1).float()
        #print('POS SHAPE',pos.shape)
        #print('POS',pos)

        #Storing other features for the node
        feat = torch.stack(
                         [
                         hit_r,
                         torch.tensor(hits.to_numpy()[:,4]),
                         torch.tensor(hits.to_numpy()[:,5]),
                         torch.tensor(hits.to_numpy()[:,6]),
                         ]
                         , dim=-1).float()

        #print ('NODE FEAT',feat)

        nhit = len(pos)

        #edge info
        #Doublet level
        d_dr = []
        d_dz = []
        d_rz = []

        for i in range(len(hit_x)-1):
            #print(hit_r[i], hit_r[i+1])
            dr = hit_r[i+1] - hit_r[i]
            d_dr.append(dr)
            #print (dr)

            #print(hit_z[i],hit_z[i+1])
            dz = hit_z[i+1] - hit_z[i]
            d_dz.append(dz)
            #print (dz)

            rz_angle = math.atan2(dz,dr)
            d_rz.append(rz_angle)
            #print (rz_angle)

        Doublet_dr = torch.FloatTensor(d_dr)
        Doublet_dz = torch.FloatTensor(d_dz)
        Doublet_rz = torch.FloatTensor(d_rz)

        e_feat = torch.stack(
                            [
                            torch.FloatTensor(d_dr),
                            torch.FloatTensor(d_dz),
                            torch.FloatTensor(d_rz),
                            ]
                            , dim = -1).float()

        #print (e_feat)

        E_feat_list = []

        for i in range(len(e_feat)):
            list_feat = e_feat[i].tolist()
            E_feat_list.append(list_feat)
            E_feat_list.append(list_feat)

        E_feat = torch.FloatTensor(E_feat_list)

        #print('EDGE FEAT SHPAE',E_feat.shape)
        #print('EDGE FEAT',E_feat)

        ###Edge label

        hitID = torch.tensor(hits.to_numpy()[:,0])
        Doublet_Pair = torch.FloatTensor(doublets.to_numpy())


        #Compare the elements?

        HitID_1 = []
        HitID_2 = []

        for i in range(len(hitID)-1):
            ID_1 = hitID[i]
            ID_2 = hitID[i+1]

            HitID_1.append(ID_1)
            HitID_2.append(ID_2)

        HID_1 = torch.FloatTensor(HitID_1)
        HID_2 = torch.FloatTensor(HitID_2)

        HID_T = torch.stack(
                            [
                            torch.FloatTensor(HitID_1),
                            torch.FloatTensor(HitID_2),
                            ]
                            , dim = -1).float()

        HitID_Truth_Match = []
        for i in range(len(HID_T)):
            #print (HID_T[i])
            for j in range(len(Doublet_Pair)):
                Check_T = torch.eq(HID_T[i],Doublet_Pair[j])
                if (Check_T[0] == torch.tensor([True])) and (Check_T[1] == torch.tensor([True])):
                    k =[1,0]

                    #print ('CHECK HIT ID',i,HID_T[i],j,Doublet_Pair[j],Check_T,k)
                    break

                else :
                    k = [0,1]
                    #print ('CHECK HIT ID',i,HID_T[i],j,Doublet_Pair[j],Check_T,k)


            HitID_Truth_Match.append(k)
            HitID_Truth_Match.append(k)

        ID_Match = torch.FloatTensor(HitID_Truth_Match)
        #print ('IDM', ID_Match)

        gr_list = []

        transform = dgl.transforms.RemoveSelfLoop()

        u = torch.arange(0,nhit-1)
        v = torch.arange(1,nhit)

        g = dgl.graph((u,v), num_nodes=nhit)
        gr = dgl.to_bidirected(g)
        gr = transform(gr)

        #print ('POS SHAPE', pos.shape)
        #print ('POS',pos)
        #print ('EN',feat)
        #print ('EDGE SHAPE', E_feat.shape)
        #print ('EDGE',E_feat)

        gr.ndata['x'] = pos[0:len(pos)+1,:]#Select [rows,cols]
        gr.ndata['en'] = feat[0:len(feat)+1,:]
        gr.edata['D_en'] = E_feat[0:len(E_feat)+1,:]
        gr.edata['label'] = ID_Match[0:len(ID_Match)+1]
        #print ('Label type', type(gr.edata['label']) )
        gr_list.append(gr)

        return {'gr' : gr_list,

                'nd' : torch.tensor([len( doublets.to_numpy() )]),

                'sc' : gr.edata['label']
            }

def collate_graphs(event_list) :

        graph = event_list[0]['gr']
        label = event_list[0]['nd']
        score = event_list[0]['sc']
        #print ('G',graph)
        #print ('L',label)
        #print ('S',score)

        #return gr_list, torch.cat(label_list), torch.cat(score)
        return graph,label,score
