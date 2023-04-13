import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch

import dgl
from dgl import backend as F

from torch.utils.data import Dataset, DataLoader, Sampler

scale_dict = {'x':    {'mean': 0.7955797898988717, 'std': 351.7879877472925},
              'y':    {'mean': -1.8853525127942654, 'std': 355.24818470948236},
              'z':    {'mean': -1.9742178380456767, 'std': 456.00292476616335},
              'v_id': {'mean': 11.667662307183546, 'std': 3.4055721162267885},
              'l_id': {'mean': 4.501973501738815, 'std': 2.1796468014394397},
              'm_id': {'mean': 671.1437562672015, 'std': 631.0579997461017}
             }


class TrackMLDataset(Dataset):
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

        hits= pd.read_csv(self.path + 'event00000%s-hits.csv'%event_idx)
        doublets = pd.read_csv(self.path + 'event00000%s-doublets.csv'%event_idx)

        hit_x = torch.tensor( ( hits.to_numpy()[:,1] - scale_dict['x']['mean'] )/scale_dict['x']['std'] )
        hit_y = torch.tensor( ( hits.to_numpy()[:,2] - scale_dict['y']['mean'] )/scale_dict['y']['std'] )
        hit_z = torch.tensor( ( hits.to_numpy()[:,3] - scale_dict['z']['mean'] )/scale_dict['z']['std'] )

        pos = torch.stack([hit_x, hit_y, hit_z], dim=-1).float()

        feat = torch.stack(
                         [
                         torch.tensor( (hits.to_numpy()[:,4] - scale_dict['v_id']['mean'])/scale_dict['v_id']['std'] ),
                         torch.tensor( (hits.to_numpy()[:,5] - scale_dict['l_id']['mean'])/scale_dict['l_id']['std'] ),
                         torch.tensor( (hits.to_numpy()[:,6] - scale_dict['m_id']['mean'])/scale_dict['m_id']['std'] )
                         ]
                         , dim=-1).float()

        nhit = len(pos)

        print('Number of Hit : ', nhit, 'Ev idx : ', event_idx)

        gr_list = []

        for i in range(0, nhit, 1000) :

            gr = dgl.knn_graph( pos[i:i+1000,:] , 5)
            gr.ndata['x'] = pos[i:i+1000,:]
            gr.ndata['en'] = feat[i:i+1000,:]

            gr_list.append(gr)




        return {'gr' : gr_list,

                'nd' : torch.log( torch.tensor([len( doublets.to_numpy() )]) )
            }



def collate_graphs(event_list) :

        n_batch = len(event_list)
        gr_list = []
        label_list = []

        for ib in range(n_batch) :


            gr_list = gr_list + event_list[ib]['gr']
            label_list.append(event_list[ib]['nd'])


        return gr_list, torch.cat(label_list)
