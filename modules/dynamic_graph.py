import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F_n

import os, sys


import dgl
from dgl import backend as F
import dgl.function as fn
from dgl.nn.pytorch import KNNGraph

from modules.mlp import build_mlp

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

# ---------------- The EdgeConv function ---------------------- #
class EdgeConv(nn.Module):

    def __init__(self,
                 in_feat_x,
                 out_feat_x,
                 in_feat_en,
                 out_feat_en,
                 batch_norm=False,
                 k_val=5):
        super(EdgeConv, self).__init__()

        self.batch_norm = batch_norm
        #self.nng = KNNGraph(k_val)

        self.k = k_val

        self.theta_en = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [4, 5, 4],\
                                  add_batch_norm = batch_norm
                                  )

        self.phi_en   = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [4, 5, 4],\
                                  add_batch_norm = batch_norm
                                  )

        self.W        = build_mlp(inputsize  = 2*in_feat_en,\
                                  outputsize = 2,\
                                  features = [4, 5, 4],\
                                  add_batch_norm = batch_norm
                                  )

        self.theta = nn.Linear(in_feat_x, out_feat_x)
        self.phi = nn.Linear(in_feat_x, out_feat_x)
        self.W = nn.Linear(2*in_feat_x, 2)
        ##20230315 change it to en as I think this is affecting the peak in the Prob...
        #self.W = nn.Linear(2*in_feat_en, 2)
        self.theta_Linear_en = nn.Linear(in_feat_en, out_feat_en)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat_x)

    def message(self, edges):
        """The message computation function.
        """

        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])

        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.src['en'])

        ##20230315 change it to en as I think this is affecting the peak in the Prob...
        W_data = torch.cat([edges.src['x'],edges.dst['x']],1)
        #W_data = torch.cat([edges.src['en'],edges.dst['en']],1)

        W_en = F_n.softmax(self.W(W_data),dim=-1)
        #print ('W_data',W_data)
        #print ('phi_en',phi_en)

        return {'edge_x': theta_x + phi_x,
                'edge_en' :  phi_en + theta_en,
                'score_en': W_en
                }

    def forward(self, g):
        """Forward computation
        """

        x_ndata_old = g.ndata['x']
        en_ndata_old = g.ndata['en']

        if not self.batch_norm:
            g.apply_edges(self.message)

            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))
            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))
            g.update_all(fn.copy_e('score_en', 'score_en'), fn.max('score_en', 'score'))

        else:
            g.apply_edges(self.message)

            g.edata['edge_x'] = self.bn(g.edata['edge_x'])
            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))
            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))
            g.update_all(fn.copy_e('score_en', 'score_en'), fn.sum('score_en', 'score'))


        x_ndata = g.ndata['x']
        en_ndata = g.ndata['en']
        x_ndata_T = self.theta(x_ndata_old)
        en_ndata_T = self.theta_Linear_en(en_ndata_old)

        g_new = g
        g_new.ndata['x'] = x_ndata + x_ndata_T
        g_new.ndata['en'] = en_ndata + en_ndata_T

        return g_new

class Dynamic_Graph_Model(nn.Module):
    def __init__(self, feature_dims_x, feature_dims_en, nclass=2):
        super(Dynamic_Graph_Model, self).__init__()
        self.n_layers = len(feature_dims_x)-1

        self.layer_list = nn.ModuleList()

        self.layer_list.append(

                        EdgeConv(in_feat_x = 3, out_feat_x = feature_dims_x[0],\
                                 in_feat_en = 4,out_feat_en = feature_dims_en[0])
         )

        for i_l in range(self.n_layers) :

            self.layer_list.append(

                        EdgeConv(in_feat_x = feature_dims_x[i_l], out_feat_x = feature_dims_x[i_l+1],\
                                 in_feat_en = feature_dims_en[i_l], out_feat_en = feature_dims_en[i_l+1])
                )

        self.latent_project = build_mlp(inputsize = sum(feature_dims_en),\
                                        outputsize = nclass,\
                                        features = [4, 6, 4]
                                        )

        self.act = nn.ReLU()

    # -- the forward function -- #
    def forward(self, g):

        with g.local_scope() :
            out_energy = []
            w_array = []
            e_array = []

            for il in range(self.n_layers+1) :#Looping Layers
                #print ('IL: ', il)
                ig = self.layer_list[il](g)
                #print ('IG: ', ig)
                e_array.append( dgl.mean_nodes(ig, feat='en')[0] ) #Already give the mean of the feat for each nodes, 4*1 per layer
                #print ('E_ARRAY: ',e_array)
                if il == int(self.n_layers):
                    w_array.append(ig.edata['score_en'])#Append the score for each edge, 2*6 per layer
                #print ('W_ARRAY: ',w_array)
            e_array = self.latent_project(torch.cat(e_array, dim=0)) #Projection to give 2*1 -> Out_energy
            #print ('E_ARRAY AFTER: ',e_array)
            out_energy.append(e_array)
            #print ('OUT_ENERGY: ',out_energy)

            w_T = torch.stack(w_array,dim=0)#Here we stack the w_array so we have 2*6*n, where n = number of layer

            #print ('W_T: ',w_T)
            W0 = torch.transpose(w_T[:,:,0],0,1) #Transpose the p(1), n*6
            W1 = torch.transpose(w_T[:,:,1],0,1)
            #print ('W0: ',W0)
            W0_T = torch.transpose(W0,0,1) #Transpose the W0 -> 6*n
            #print ('W0_T: ',W0_T)
            AVG_W0 = torch.mean(W0,1) #Get the mean of p(1) , (n*6)/n => 6*1 => matching number of edges
            AVG_W0_T = torch.mean(W0_T,1) # Give (6*n)/6 => n*1 => Didn't match the number of edges
            #print ('AVG W0: ',AVG_W0)
            #print ('AVG W0_T: ',AVG_W0_T)
            AVG_W1 = torch.mean(W1,1)
            AVG_W = torch.stack((AVG_W0,AVG_W1),dim=1) #Put back the p(1) and p(0) => 2*6
            #print ('AVG_W: ',AVG_W)

            out_tensor = torch.cat( out_energy )
            #print ('OUT_TEN: ', out_tensor)

            Out_Fn = self.act(out_tensor).mean().reshape(1,)
            #print ('OUT_FN: ', Out_Fn)#Final onput = 1 number = number of edges

            Out_W = AVG_W #This is the Pred_score in the training code
            return Out_Fn, Out_W
