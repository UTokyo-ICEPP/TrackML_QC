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

# ----- the lorentz layer ------------------- #
class LGLayer(nn.Module):
    
    def __init__(self,
                 in_feat_x,
                 out_feat_x,
                 in_feat_h,
                 out_feat_h,
                 hidden_m,
                 batch_norm=False, device=''):
        super(LGLayer, self).__init__()
        
        self.batch_norm = batch_norm

        self.dev=device

        
        self.phi_x = build_mlp(inputsize  = hidden_m,\
                                  outputsize = 1,\
                                  features = [6, 12, 6],\
                                  add_batch_norm = batch_norm
                                  )
        
        self.phi_e   = build_mlp(inputsize  = 2*in_feat_x + 2*in_feat_h,\
                                  outputsize = hidden_m,\
                                  features = [6, 12, 6],\
                                  add_batch_norm = batch_norm
                                  )
        
        self.phi_h   = build_mlp(inputsize  = in_feat_h + hidden_m,\
                                  outputsize = out_feat_h,\
                                  features = [6, 12, 6],\
                                  add_batch_norm = batch_norm
                                  )
        
        self.phi_m   = build_mlp(inputsize  = hidden_m,\
                                  outputsize = 1,\
                                  features = [6, 12, 6],\
                                  add_batch_norm = batch_norm
                                  )
        
        self.map_x =  nn.Parameter( torch.ones(in_feat_x, out_feat_x).to(self.dev) )
        nn.init.uniform_(self.map_x)
        
        self.map_h =  nn.Parameter( torch.ones(in_feat_h, out_feat_h).to(self.dev) )
        nn.init.uniform_(self.map_h)


        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat_x)
            
        metric = -1 * torch.eye(4).to(self.dev)
        metric[0][0] = 1.
        self.metric = metric.float()

    def psi(self, x) : 
        return x.sign() * torch.log( torch.abs(x) + 1 )
    
    def dot4(self, x, y) : 
        return torch.einsum('cap,ab,cbp->cp', x, self.metric, y)
    
    def edge_fn(self, edges):
        """The message computation function.
        """
        
        xi, xj = edges.dst['x'], edges.src['x']

        
        m_ij = torch.cat([edges.dst['h'], edges.src['h'], 
                          self.psi( self.dot4(xi, xj) ),
                          self.psi( self.dot4( (xi - xj), (xi - xj) ) )], dim=-1)
        
        m_ij = self.phi_e(m_ij)
        w_ij = torch.sigmoid( self.phi_m(m_ij) )
        
        m_x = torch.einsum('cb,cab->cab', self.phi_x(m_ij), xj)
        m_h = w_ij * m_ij
        
        
        return{'m_x' : m_x, 
               'm_h' : m_h}
    
    def node_fn(self, nodes):
        """The node aggregration function.
        """
        
        h_new = torch.einsum('ni,ij->nj',  nodes.data['h'], self.map_h) + self.phi_h(
            torch.cat([nodes.data['h'],  torch.sum(nodes.mailbox['m_h'], dim=1)], dim=-1)
            )
        
        x_new = nodes.data['x'] + 0.5 * torch.sum(nodes.mailbox['m_x'], dim=1)
        
        x_new = torch.einsum('nxi,ij->nxj',  x_new, self.map_x)
        return{'x' : x_new, 
               'h' : h_new
               }

    def forward(self, g):
        """Forward computation
        """
        x = g.ndata['x']
        h = g.ndata['h']
        
        g.update_all(self.edge_fn, self.node_fn)
        
        return g

class LorentzGroup_Model(nn.Module):
    def __init__(self, init_dim, feature_dims_x, feature_dims_h, feature_dims_m, device):
        super(LorentzGroup_Model, self).__init__()

        self.n_layers = len(feature_dims_x)-1

        self.dev = device

        
        self.layer_list = []

        self.layer_list.append( 

                        LGLayer(in_feat_x = init_dim[0], out_feat_x = feature_dims_x[0],\
                                in_feat_h = init_dim[1], out_feat_h = feature_dims_h[0],\
                                hidden_m= feature_dims_m[0], device=self.dev)
         )


        for i_l in range(self.n_layers) : 

            self.layer_list.append(

                        LGLayer(in_feat_x = feature_dims_x[i_l], out_feat_x = feature_dims_x[i_l+1],\
                                in_feat_h = feature_dims_h[i_l], out_feat_h = feature_dims_h[i_l+1],\
                                     hidden_m= feature_dims_m[i_l+1], device=self.dev
                                     )
                )

        self.layers = nn.ModuleList(self.layer_list)

        self.latent_project = build_mlp(inputsize = feature_dims_h[-1],\
                                        outputsize = 2,\
                                        features = [4, 6, 4]
                                        )


    def forward(self, g):
        
        with g.local_scope() : 
            out_energy = []

            
            e_array = []
            for il in self.layers : 

                g = il(g)              
                

            gr_list = dgl.unbatch(g)
            
            for ig in gr_list : 
                
                out_energy.append(dgl.mean_nodes(graph=ig, feat='h'))
                
            
            y = self.latent_project(torch.stack(out_energy, dim=0)[:,0,:])
            
            return torch.softmax(y, dim=-1)