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

        #This is for the edge convolution
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

        #This is for edge classification
        #self.W        = build_mlp(inputsize  = 2*in_feat_en,\
        #                          outputsize = 2,\
        #                          features = [4, 5, 4],\
        #                          add_batch_norm = batch_norm
        #                          )

        #Updated at 12th April, as we changed the activation function to sigmoid
        #In order to get 1 output ranging from 0 to 1
        self.W2        = build_mlp(inputsize  = 2*out_feat_x,\
                                  outputsize = 1,\
                                  features = [4, 5, 4],\
                                  add_batch_norm = batch_norm
                                  )


        self.theta = nn.Linear(in_feat_x, out_feat_x)
        self.phi = nn.Linear(in_feat_x, out_feat_x)
        #self.W = nn.Linear(2*in_feat_x, 2)

        # 2*out_feat_x = matching the shape coming from theta and phi
        self.W2 = nn.Linear(2*out_feat_x, 1)
        #print ("Self.W2",self.W2)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat_x)

    def message(self, edges):
        """The message computation function.
        """

        #theta, phi: Updating the edges in the edge conv steps
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        #print ("theta_x",theta_x.shape)
        #print ("phi_x",phi_x.shape)

        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.src['en'])
        #print ("theta_en",theta_x.shape)
        #print ("phi_en",phi_x.shape)

        #W_data = torch.cat([edges.src['x'],edges.dst['x']],1)

        #Edge classification, using theta and phi as an input
        W_data_2 = torch.cat([theta_x,phi_x],1)

        #W_en = F_n.softmax(self.W(W_data),dim=-1)

        #Transforms by self.W2 and use sigmoid to get the edge scores
        W_en = torch.sigmoid(self.W2(W_data_2))#,dim=-1)

        return {'edge_x': theta_x + phi_x,
                'edge_en' :  phi_en + theta_en,
                'score_en': W_en
                }

    def forward(self, g):
        """Forward computation
        """
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


        #'x': (x,y,z) for the hits
        x_ndata = g.ndata['x']
        # 'en': Whatever else features coming from the hit input
        en_ndata = g.ndata['en']

        g_new = g

        g_new.ndata['x'] = x_ndata
        g_new.ndata['en'] = en_ndata

        #print("NODE data X:",g_new.ndata['x'].shape,x_ndata.shape)
        #print("NODE data EN:",g_new.ndata['en'].shape,en_ndata.shape)

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
            out_W = []
            #print("dynamic_graph.py: g",g)
            #print("dynamic_graph.py: batch_size",g.batch_size)
            graph_list = dgl.unbatch(g)
            #print("dynamic_graph.py: graph_list",graph_list)

            for ig in graph_list :
                w_array = []
                #e_array = []
                #print("dynamic_graph.py: ig",ig)

                #We need to extract the score from the last layer
                for il in range(self.n_layers+1) :#Looping Layers

                    ilg = self.layer_list[il](ig)

                    #print (ilg.edata['score_en'])
                    if il == int(self.n_layers):
                        w_array.append(ilg.edata['score_en'])#Only store the score at the output layer
                        #print ('W_ARRAY: ',w_array)

                #Transforms the scores to match the taget format
                W0 = w_array[0][:,0]

                #Collect all the scores from graphs in graph list
                out_W.append(W0)

            #Make it as a tensor
            Out_WFn = torch.cat(out_W)
            #print ("Out_WFn",Out_WFn)
            return Out_WFn
