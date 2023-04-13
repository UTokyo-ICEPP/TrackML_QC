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

start_N = 359403
end_N = 359503

df_h = pd.read_csv('/data/wachan/ds100/event000001000-hits.csv')
df_d = pd.read_csv('/data/wachan/ds100/event000001000-doublets.csv')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

print('cuda_device : ', cuda_device)

from modules.OneTrackDataloader import  OneTrackDataset, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model

path = '/data/wachan/ds100/OneTrackSamples/evt1000_Nearby_Hit_v4_Random/'

#50%
data_set_test = OneTrackDataset(path, num_start=359403, num_end=449260)

#100%
#data_set_test = OneTrackDataset(path, num_start=718803, num_end=898516)

#20%
#data_set_test = OneTrackDataset(path, num_start=143763, num_end=179706)

#Test
#data_set_test = OneTrackDataset(path, num_start=start_N, num_end=end_N)

test_loader = DataLoader(data_set_test, batch_size=128, shuffle=False,collate_fn=collate_graphs, num_workers=0)

model = Dynamic_Graph_Model(feature_dims_x = [4,32,16,3], feature_dims_en = [4,32,16,4])
model_name = 'model_DynamicGraphTrack_Score_NH_v4_50p_batch_size_128.pt'
out_name = 'PredictionFile_DynamicGraphTrack_Score_NH_v4_batch_size_128.h5'

model.to(cuda_device)

model.load_state_dict(torch.load(model_name))

param_numb = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters : ', param_numb)

model.eval()

truth_scores_Match, predicted_scores_Match, truth_scores_MisMatch, predicted_scores_MisMatch = [], [], [], []
T_QS_Match,T_QS_MisMatch,P_QS_Match,P_QS_MisMatch = [], [], [], []
Triplet_Sum = {}

with tqdm(test_loader, ascii=True) as tq:
    count = start_N
    for graph, label, score in tq:

        print (count)

        T_QS_Match_list,T_QS_MisMatch_list,P_QS_Match_list,P_QS_MisMatch_list = [], [], [], []
        score = score.to(cuda_device)
        pred_score = model(graph[0].to(cuda_device))[1]
        #print (graph[0].ndata['x'][0][0])

        for i in range(len(pred_score)):
            #print (i)
            truth_scores_Match.append(score[i][0].item())
            truth_scores_MisMatch.append(score[i][1].item())
            predicted_scores_Match.append(pred_score[i][0].item())
            predicted_scores_MisMatch.append(pred_score[i][1].item())
            T_QS_Match_list.append(score[i][0].item())
            T_QS_MisMatch_list.append(score[i][1].item())
            P_QS_Match_list.append(pred_score[i][0].item())
            P_QS_MisMatch_list.append(pred_score[i][1].item())
            #print (i)
            #if (pred_score[i][0] > 0.85):
                #print ()
                #print ('Less than 0.9995, with pred score: ', pred_score[i][0].item(), ', target score: ', score[i][0].item(), ', ', score[i][1].item())

        del graph; del score; del pred_score;

        #print (T_QS_Match_list)
        #print (sum(PScore_list)/2)
        if len(T_QS_Match_list) == 6:
            T_QS_Match.append(sum(T_QS_Match_list))
            T_QS_MisMatch.append(sum(T_QS_MisMatch_list))
            P_QS_Match.append(sum(P_QS_Match_list))
            P_QS_MisMatch.append(sum(P_QS_MisMatch_list))

            print (P_QS_Match)

            hits= pd.read_csv(path + 'OneTrackSample00%s-hits.csv'%count)
            doublets = pd.read_csv(path + 'OneTrackSample00%s-doublets.csv'%count)
            #print (doublets)
            print (doublets['start'][0])
            #print (hits)
            
            #Need to implement the formula to get the strength
            Q_Strength = 0

            Q_Pair_1 = str(doublets['start'][0])+'_'+str(doublets['start'][1])+'_'+str(doublets['start'][2])
            Q_Pair_2 = str(doublets['end'][0])+'_'+str(doublets['end'][1])+'_'+str(doublets['end'][2])

            #print (Q_Pair_1,Q_Pair_2,Q_Strength)
            Triplet_Pair = "{},{}".format(Q_Pair_1,Q_Pair_2)
            #print (Triplet_Pair)
            Triplet_Sum[Triplet_Pair] = Q_Strength
            #print (Triplet_Sum)

            #if count == 359405:
                #exit()

        count += 1
#print (len(PScore_All))

hf = h5py.File(out_name, 'w')

hf.create_dataset('truth_scores_Match', data=np.array(truth_scores_Match))
hf.create_dataset('truth_scores_MisMatch', data=np.array(truth_scores_MisMatch))
hf.create_dataset('predicted_scores_Match', data=np.array(predicted_scores_Match))
hf.create_dataset('predicted_scores_MisMatch', data=np.array(predicted_scores_MisMatch))
hf.create_dataset('T_QS_Match', data=np.array(T_QS_Match))
hf.create_dataset('T_QS_MisMatch', data=np.array(T_QS_MisMatch))
hf.create_dataset('P_QS_Match', data=np.array(P_QS_Match))
hf.create_dataset('P_QS_MisMatch', data=np.array(P_QS_MisMatch))

hf.close()

