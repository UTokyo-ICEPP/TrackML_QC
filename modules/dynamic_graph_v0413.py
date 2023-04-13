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
                 batch_norm=False):
        super(EdgeConv, self).__init__()

        self.batch_norm = batch_norm

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

        self.theta = nn.Linear(in_feat_x, out_feat_x)
        self.phi = nn.Linear(in_feat_x, out_feat_x)

        #This is the MLP we used to predict the score for the edges within each EdgeConv circule
        self.W         = build_mlp(inputsize  = (2*out_feat_x)+(2*out_feat_en)+3,\
                                  outputsize = 1,\
                                  features = [4, 5, 4],\
                                  add_batch_norm = batch_norm
                                  )

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat_x)

    def message(self, edges):
        """The message computation function.
        """

        #theta, phi: Updating the edges in the edge conv steps
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.src['en'])

        #Score_input: Making a Tensor as the input to pass the MLP (self.W) for score prediction
        Score_input = torch.cat([theta_x, phi_x, theta_en, phi_en, edges.data['D_en']],1)

        #The Score is given by the sigmoid function (ranging from 0 to 1)
        Score = torch.sigmoid(self.W(Score_input))

        return {'edge_x': theta_x + phi_x,
                'edge_en' : phi_en + theta_en,
                'score': Score
                }

    def forward(self, g):
        """Forward computation
        """
        if not self.batch_norm:
            g.apply_edges(self.message)

            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))
            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))
            g.update_all(fn.copy_e('score', 'score'), fn.mean('score', 'score_n'))

        else:
            g.apply_edges(self.message)

            g.edata['edge_x'] = self.bn(g.edata['edge_x'])
            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))
            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))
            g.update_all(fn.copy_e('score', 'score'), fn.mean('score', 'score_n'))

        return g

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

        self.act = nn.ReLU()

    # -- the forward function -- #
    def forward(self, g):

        with g.local_scope() :
            score_array = []
            graph_list = dgl.unbatch(g)

            for ig in graph_list :
                for il in range(self.n_layers+1) :

                    ilg = self.layer_list[il](ig)

                #Collect the score for each graph only from the last layer
                score_array.append(ig.edata['score'][:,0])

            #Make a Tensor which contain all the scores for each graph within the graphlist
            Score_Tensor = torch.cat(score_array)

            return Score_Tensor
